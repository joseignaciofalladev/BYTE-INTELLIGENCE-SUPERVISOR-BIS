// BYTE INTELLIGENCE SUPERVISOR (BIS) - Prototype implementation
// C++17, single-file demo/prototype
// Compile: g++ -std=c++17 bis.cpp -pthread -O2 -o bis_demo
// Run: ./bis_demo

#include <bits/stdc++.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <queue>
#include <optional>
#include <sstream>
using namespace std::chrono_literals;

// Config & types
struct Config {
    double decisionHz = 20.0;            // Decision tick frequency (Hz)
    size_t maxActionsPerTick = 8;        // Limit actions executed per tick
    size_t maxBytesMovePerTick = 10 * 1024 * 1024; // 10 MB simulated budget per tick
    double sigmoidK = 1.0;               // scoring logistic factor
    double S0 = 64.0 * 1024.0;           // size denom for scoring (bytes)

    // scoring weights (beta)
    double betaU = 2.0;
    double betaP = 3.0;
    double betaCd = 1.5;
    double betaBin = 1.2;
    double betaS = 1.0;
    double betaR = 3.0;
    double betaD = 2.0;
} cfg;

using Timestamp = uint64_t;
static Timestamp NowMs() {
    return (Timestamp)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Utility: simple thread-safe logger
class Logger {
    std::mutex m_;
public:
    enum Level { Info, Warn, Error };
    void log(Level l, const std::string &s) {
        std::lock_guard<std::mutex> lk(m_);
        const char* pre = (l==Info?"[INFO] ": (l==Warn?"[WARN] ":"[ERR] "));
        std::cout << pre << s << std::endl;
    }
} g_log;

// Resource model & telemetry
enum class ResourceType { Texture, Mesh, Audio, Animation, Latent, Shader, Probe, Other };

struct ResourceID {
    uint64_t id;
    bool operator==(ResourceID const& o) const { return id == o.id; }
};

struct ResourceKeyHash { size_t operator()(ResourceID const& k) const noexcept { return std::hash<uint64_t>()(k.id); } };

struct ResourceSummary {
    ResourceID rid;
    ResourceType type;
    size_t sizeBytes;           // current resident size
    size_t residentBytes;       // resident in VRAM/RAM
    double lastAccessSec;       // seconds since last access (float)
    uint32_t accessCountWindow; // number of accesses in sliding window
    double decodeCostMs;        // estimated cost to decode/regenerate
    double bandwidthCostKB;     // cost to stream back (KB)
    double perceptualImportance; // 0..1 from PFG/XFR
    double riskScore;           // 0..1 (NERVA warnings etc)
    double dedupePotential;     // 0..1; >0.8 means good candidate to dedupe
    Timestamp lastAccessTs;     // last access ms
    bool pinned = false;        // hero / pinned resources
    std::string humanName;      // debugging name
    // derived / runtime
    double bvsScore = 0.0;      // computed Byte Value Score
};

////////////////////////////////////////////////////////////////////////////////
// Resource Table (RT)
////////////////////////////////////////////////////////////////////////////////
class ResourceTable {
    std::unordered_map<ResourceID, ResourceSummary, ResourceKeyHash> table_;
    std::mutex m_;
public:
    void upsert(const ResourceSummary& s) {
        std::lock_guard<std::mutex> lk(m_);
        table_[s.rid] = s;
    }
    bool get(ResourceID id, ResourceSummary &out) {
        std::lock_guard<std::mutex> lk(m_);
        auto it = table_.find(id);
        if (it==table_.end()) return false;
        out = it->second;
        return true;
    }
    void updateAccess(ResourceID id, size_t bytes) {
        std::lock_guard<std::mutex> lk(m_);
        auto it = table_.find(id);
        if (it==table_.end()) return;
        it->second.lastAccessTs = NowMs();
        it->second.accessCountWindow++;
        it->second.residentBytes = bytes;
        it->second.lastAccessSec = 0.0; // reset; consumers can compute deltas
    }
    std::vector<ResourceSummary> snapshotCandidates() {
        std::lock_guard<std::mutex> lk(m_);
        std::vector<ResourceSummary> out;
        out.reserve(table_.size());
        for (auto &p : table_) out.push_back(p.second);
        return out;
    }
    void setPinned(ResourceID id, bool pinned) {
        std::lock_guard<std::mutex> lk(m_);
        auto it = table_.find(id);
        if (it!=table_.end()) it->second.pinned = pinned;
    }
    void markResident(ResourceID id, size_t residentBytes) {
        std::lock_guard<std::mutex> lk(m_);
        auto it = table_.find(id);
        if (it!=table_.end()) it->second.residentBytes = residentBytes;
    }
    void remove(ResourceID id) {
        std::lock_guard<std::mutex> lk(m_);
        table_.erase(id);
    }
};

////////////////////////////////////////////////////////////////////////////////
// Telemetry Agents (simulated) - in a real integration these are push hooks
////////////////////////////////////////////////////////////////////////////////
class TelemetryAgent {
    ResourceTable &rt_;
    std::atomic<uint64_t> nextId_{1};
public:
    TelemetryAgent(ResourceTable &rt): rt_(rt) {}
    ResourceID create(ResourceType t, size_t sizeBytes, double decodeCostMs, const std::string &name) {
        ResourceID id{ nextId_.fetch_add(1) };
        ResourceSummary s;
        s.rid = id;
        s.type = t;
        s.sizeBytes = sizeBytes;
        s.residentBytes = sizeBytes;
        s.lastAccessTs = NowMs();
        s.accessCountWindow = 0;
        s.decodeCostMs = decodeCostMs;
        s.bandwidthCostKB = (double)sizeBytes / 1024.0;
        s.perceptualImportance = 0.5;
        s.riskScore = 0.0;
        s.dedupePotential = 0.0;
        s.humanName = name;
        rt_.upsert(s);
        return id;
    }
    void touch(ResourceID id, size_t residentBytes) {
        rt_.updateAccess(id, residentBytes);
    }
    void setPerceptual(ResourceID id, double p) {
        ResourceSummary s;
        if (rt_.get(id, s)) {
            s.perceptualImportance = p;
            rt_.upsert(s);
        }
    }
    void setDedupe(ResourceID id, double d) {
        ResourceSummary s;
        if (rt_.get(id, s)) {
            s.dedupePotential = d;
            rt_.upsert(s);
        }
    }
    void setRisk(ResourceID id, double r) {
        ResourceSummary s;
        if (rt_.get(id, s)) {
            s.riskScore = r;
            rt_.upsert(s);
        }
    }
    void pin(ResourceID id, bool pinned) { rt_.setPinned(id, pinned); }
};

////////////////////////////////////////////////////////////////////////////////
// Scoring engine: Byte Value Score (BVS)
////////////////////////////////////////////////////////////////////////////////
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-cfg.sigmoidK * x));
}

double scoreResource(const ResourceSummary &r) {
    // map simple metrics into a raw score
    double U = (double)r.accessCountWindow; // normalized later
    double P = r.perceptualImportance;
    double Cd = r.decodeCostMs;
    double Bin = r.bandwidthCostKB;
    double S = (double)r.sizeBytes;
    double R = r.riskScore;
    double D = r.dedupePotential;

    // simple normalization heuristics
    double Unorm = U / (U + 10.0);   // saturates beyond 10 accesses in window
    double raw = cfg.betaU * Unorm + cfg.betaP * P
                 - cfg.betaCd * std::log(Cd + 1.0)
                 - cfg.betaBin * (Bin/(Bin+10.0))
                 - cfg.betaS * (S/(S + cfg.S0))
                 - cfg.betaR * R
                 + cfg.betaD * D;
    double bvs = sigmoid(raw); // map to 0..1
    return bvs;
}

////////////////////////////////////////////////////////////////////////////////
// Action model
////////////////////////////////////////////////////////////////////////////////
enum class ActionType { EvictToDisk, TranscodeLowerLOD, CompressInPlace, Pin, Prefetch, ConsolidateDuplicate };

struct Action {
    ActionType type;
    ResourceID target;
    size_t expectedBytesFreed = 0;
    double expectedCostMs = 0.0;
    std::string reason;
};

////////////////////////////////////////////////////////////////////////////////
// Executor (threaded) - applies actions safely with rollback handles
////////////////////////////////////////////////////////////////////////////////

struct Checkpoint {
    ResourceID target;
    // In a real system, would store metadata to restore (pointer, resident bytes, small snapshots)
    size_t previousResidentBytes;
    Timestamp createdAt;
};

class Executor {
    std::mutex queueM;
    std::condition_variable cv;
    std::queue<Action> workQ;
    std::atomic<bool> running{true};
    std::thread worker;
    ResourceTable &rt_;
    std::mutex cpM;
    std::vector<Checkpoint> checkpoints; // recent checkpoints
    std::mutex logM;
public:
    Executor(ResourceTable &rt): rt_(rt) {
        worker = std::thread([this]{ this->loop(); });
    }
    ~Executor() {
        running = false;
        cv.notify_all();
        if (worker.joinable()) worker.join();
    }
    void enqueue(Action a) {
        {
            std::lock_guard<std::mutex> lk(queueM);
            workQ.push(a);
        }
        cv.notify_one();
    }
    void loop() {
        while (running) {
            Action a;
            {
                std::unique_lock<std::mutex> lk(queueM);
                cv.wait_for(lk, 100ms, [this]{ return !workQ.empty() || !running; });
                if (!running) break;
                if (workQ.empty()) continue;
                a = workQ.front(); workQ.pop();
            }
            applyActionSafe(a);
        }
    }

    // Basic simulation of action application with checkpoint/rollback
    void applyActionSafe(const Action &a) {
        // fetch resource summary
        ResourceSummary s;
        if (!rt_.get(a.target, s)) {
            g_log.log(Logger::Warn, "Executor: resource not found for action");
            return;
        }
        if (s.pinned && a.type != ActionType::Pin) {
            g_log.log(Logger::Info, "Executor: skip action because resource pinned: " + s.humanName);
            return;
        }

        // create checkpoint
        Checkpoint cp{a.target, s.residentBytes, NowMs()};
        {
            std::lock_guard<std::mutex> lk(cpM);
            checkpoints.push_back(cp);
            if (checkpoints.size() > 200) checkpoints.erase(checkpoints.begin()); // cap
        }

        bool success = false;
        // Simulated action application & cost
        if (a.type==ActionType::EvictToDisk) {
            g_log.log(Logger::Info, "Executing EvictToDisk for " + s.humanName + " expected free " + std::to_string(a.expectedBytesFreed));
            // TODO: call ASTRA to write page to disk/compressed store
            std::this_thread::sleep_for(std::chrono::milliseconds((int)std::max(1.0, a.expectedCostMs)));
            // simulate success and free bytes
            rt_.markResident(a.target, std::max<size_t>(0, (int)s.residentBytes - (int)a.expectedBytesFreed));
            success = true;
        } else if (a.type==ActionType::CompressInPlace) {
            g_log.log(Logger::Info, "CompressInPlace for " + s.humanName);
            // TODO: replace with real compression (fast) and update resident size
            std::this_thread::sleep_for(std::chrono::milliseconds((int)std::max(1.0, a.expectedCostMs)));
            size_t newSize = (size_t)((double)s.residentBytes * 0.6); // mock 40% saving
            rt_.markResident(a.target, newSize);
            success = true;
        } else if (a.type==ActionType::TranscodeLowerLOD) {
            g_log.log(Logger::Info, "TranscodeLowerLOD for " + s.humanName);
            // TODO: produce lower LOD, update mapping
            std::this_thread::sleep_for(std::chrono::milliseconds((int)std::max(1.0, a.expectedCostMs)));
            size_t newSize = (size_t)((double)s.residentBytes * 0.5);
            rt_.markResident(a.target, newSize);
            success = true;
        } else if (a.type==ActionType::Pin) {
            g_log.log(Logger::Info, "Pin resource " + s.humanName + " -> " + a.reason);
            rt_.setPinned(a.target, true);
            success = true;
        } else if (a.type==ActionType::Prefetch) {
            g_log.log(Logger::Info, "Prefetch " + s.humanName);
            // TODO: ensure resource resident (simulate streaming)
            std::this_thread::sleep_for(50ms);
            rt_.markResident(a.target, s.sizeBytes);
            success = true;
        } else if (a.type==ActionType::ConsolidateDuplicate) {
            g_log.log(Logger::Info, "Consolidate duplicate " + s.humanName);
            // TODO: update indirection tables; here we simulate a small free
            std::this_thread::sleep_for(20ms);
            rt_.markResident(a.target, (size_t)((double)s.residentBytes * 0.8));
            success = true;
        }

        // check for anomalies via a simulated NERVA check (stub)
        if (!success) {
            g_log.log(Logger::Warn, "Executor: action failed, rolling back");
            rollbackLast(cp.target);
        } else {
            g_log.log(Logger::Info, "Executor: action succeeded: " + s.humanName);
            // record forensic log entry - simplified
            logForensics(a, cp);
        }
    }

    void rollbackLast(ResourceID rid) {
        std::lock_guard<std::mutex> lk(cpM);
        for (auto it = checkpoints.rbegin(); it != checkpoints.rend(); ++it) {
            if (it->target.id == rid.id) {
                // restore state
                ResourceSummary s;
                if (rt_.get(rid, s)) {
                    rt_.markResident(rid, it->previousResidentBytes);
                }
                g_log.log(Logger::Warn, "Rollback applied for resource id=" + std::to_string(rid.id));
                return;
            }
        }
    }

    void logForensics(const Action &a, const Checkpoint &cp) {
        std::lock_guard<std::mutex> lk(logM);
        std::ostringstream ss;
        ss << "{forensic: {ts:" << NowMs() << ", action:\"" << (int)a.type << "\", target:" << a.target.id
           << ", prevSize:" << cp.previousResidentBytes << "}}";
        g_log.log(Logger::Info, ss.str());
    }
};

////////////////////////////////////////////////////////////////////////////////
// Policy Engine - greedy planner
////////////////////////////////////////////////////////////////////////////////

class PolicyEngine {
public:
    // produce plan given candidate resources sorted by BVS ascending (low BVS -> good target)
    static std::vector<Action> producePlan(const std::vector<ResourceSummary> &candidates, size_t budgetBytes, size_t maxActions) {
        std::vector<Action> plan;
        size_t used = 0;
        for (auto const &r : candidates) {
            if (r.pinned) continue; // skip pinned
            if (plan.size() >= maxActions) break;
            // simple policy: choose action based on type and size
            Action a;
            a.target = r.rid;
            a.expectedCostMs = std::max(1.0, r.decodeCostMs * 0.5);
            // heuristics
            if (r.dedupePotential > 0.7) {
                a.type = ActionType::ConsolidateDuplicate;
                a.expectedBytesFreed = (size_t)(r.residentBytes * 0.15);
                a.reason = "dedupe";
            } else if (r.perceptualImportance < 0.2) {
                // low importance: evict or transcode
                if (r.sizeBytes > 512*1024) {
                    a.type = ActionType::EvictToDisk;
                    a.expectedBytesFreed = r.residentBytes;
                    a.reason = "lowP + large";
                } else {
                    a.type = ActionType::TranscodeLowerLOD;
                    a.expectedBytesFreed = (size_t)(r.residentBytes * 0.5);
                    a.reason = "lowP small -> transcode";
                }
            } else if (r.perceptualImportance < 0.5) {
                a.type = ActionType::CompressInPlace;
                a.expectedBytesFreed = (size_t)(r.residentBytes * 0.3);
                a.reason = "moderateP -> compress";
            } else {
                continue; // too important
            }
            if (used + a.expectedBytesFreed > budgetBytes) continue;
            used += a.expectedBytesFreed;
            plan.push_back(a);
        }
        return plan;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Forensics / PASS lesson emitter (simple)
////////////////////////////////////////////////////////////////////////////////
struct Lesson {
    std::string lessonID;
    Timestamp ts;
    ResourceID rid;
    std::string symptom;
    std::string suggestion;
    double confidence;
    std::string evidence;
};

class PassBridge {
    std::mutex m_;
public:
    void emitLesson(const Lesson &L) {
        std::lock_guard<std::mutex> lk(m_);
        // In real integration: serialize and send to PASS cooker pipeline (CI)
        std::ostringstream ss;
        ss << "{lessonID:\"" << L.lessonID << "\",ts:" << L.ts << ",rid:" << L.rid.id
           << ",symptom:\"" << L.symptom << "\",suggest:\"" << L.suggestion << "\",conf:" << L.confidence << "}";
        g_log.log(Logger::Info, "PASS lesson emitted: " + ss.str());
    }
};

////////////////////////////////////////////////////////////////////////////////
// BIS core: DecisionTick loop & orchestration
////////////////////////////////////////////////////////////////////////////////
class BIS {
    ResourceTable &rt_;
    Executor &exec_;
    PassBridge &pass_;
    std::atomic<bool> running{true};
    std::thread decisionThread;
public:
    BIS(ResourceTable &rt, Executor &exec, PassBridge &pass): rt_(rt), exec_(exec), pass_(pass) {
        decisionThread = std::thread([this]{ this->decisionLoop(); });
    }
    ~BIS() {
        running = false;
        if (decisionThread.joinable()) decisionThread.join();
    }

    void decisionLoop() {
        using namespace std::chrono;
        double hz = cfg.decisionHz;
        auto period = duration_cast<milliseconds>(duration<double>(1.0/hz));
        while (running) {
            auto t0 = steady_clock::now();
            runDecisionTick();
            auto t1 = steady_clock::now();
            auto elapsed = duration_cast<milliseconds>(t1 - t0);
            if (elapsed < period) std::this_thread::sleep_for(period - elapsed);
        }
    }

    void runDecisionTick() {
        // 1. snapshot candidates
        auto candidates = rt_.snapshotCandidates();
        if (candidates.empty()) return;

        // 2. compute BVS for each and sort ascending (lower score = better eviction target)
        for (auto &c : candidates) c.bvsScore = scoreResource(c);
        std::sort(candidates.begin(), candidates.end(), [](auto &a, auto &b){
            return a.bvsScore < b.bvsScore;
        });

        // 3. decide budget for this tick based on free memory heuristics (simulated)
        size_t budgetBytes = cfg.maxBytesMovePerTick;
        // adapt budget: if many high-perceptual items, reduce budget to avoid visible change
        double avgP = 0.0;
        for (size_t i=0;i<candidates.size();++i) avgP += candidates[i].perceptualImportance;
        avgP /= std::max<double>(1.0, (double)candidates.size());
        if (avgP > 0.6) budgetBytes /= 2;

        // 4. produce plan
        auto plan = PolicyEngine::producePlan(candidates, budgetBytes, cfg.maxActionsPerTick);

        // 5. simulate & filter (cheap sim: ensure not evicting recently accessed)
        std::vector<Action> safePlan;
        Timestamp now = NowMs();
        for (auto &a : plan) {
            ResourceSummary s;
            if (!rt_.get(a.target, s)) continue;
            // do not evict if last access less than 200ms (hot)
            if (now - s.lastAccessTs < 200) continue;
            // skip if pinned
            if (s.pinned) continue;
            // skip if risk flagged
            if (s.riskScore > 0.7) {
                // emit lesson to PASS: resource risky, do not touch
                Lesson L;
                L.lessonID = "L-" + std::to_string(now) + "-" + std::to_string(s.rid.id);
                L.ts = now;
                L.rid = s.rid;
                L.symptom = "High risk flagged - avoid eviction";
                L.suggestion = "Investigate asset cook/seed";
                L.confidence = 0.9;
                L.evidence = s.humanName;
                pass_.emitLesson(L);
                continue;
            }
            safePlan.push_back(a);
        }

        // 6. enqueue safe plan actions to executor
        for (auto &a : safePlan) {
            exec_.enqueue(a);
        }

        // 7. emit light telemetry log
        {
            std::ostringstream ss;
            ss << "DecisionTick: candidates=" << candidates.size() << " plan=" << safePlan.size()
               << " avgP=" << avgP;
            g_log.log(Logger::Info, ss.str());
        }
    }
};

////////////////////////////////////////////////////////////////////////////////
// Demo: simulate resources, touches and BIS behavior
////////////////////////////////////////////////////////////////////////////////

int main() {
    g_log.log(Logger::Info, "BIS prototype starting...");

    ResourceTable rt;
    TelemetryAgent tele(rt);

    // create a mix of resources
    std::vector<ResourceID> all;
    all.push_back(tele.create(ResourceType::Texture, 4*1024*1024, 10.0, "albedo_city_1K"));
    all.push_back(tele.create(ResourceType::Texture, 2*1024*1024, 6.0, "brick_wall_1K"));
    all.push_back(tele.create(ResourceType::Mesh,    800*1024,  2.0, "car_model_lod0"));
    all.push_back(tele.create(ResourceType::Mesh,    200*1024,  1.0, "bench_mesh"));
    all.push_back(tele.create(ResourceType::Latent,  8*1024,    0.5, "tessera_tile_45"));
    all.push_back(tele.create(ResourceType::Shader,  64*1024,   0.2, "pbr_material_shader"));
    all.push_back(tele.create(ResourceType::Texture, 6*1024*1024, 12.0,"hero_character_diffuse"));
    all.push_back(tele.create(ResourceType::Texture, 512*1024,  3.0, "grass_albedo"));

    // mark hero resource pinned
    tele.pin(all[6], true);

    // adjust some perceptual importance & dedupe potential
    tele.setPerceptual(all[0], 0.8); // city albedo high P
    tele.setPerceptual(all[7], 0.15); // grass low P
    tele.setDedupe(all[1], 0.9); // brick dedupe candidate

    // create executor & pass bridge & BIS
    PassBridge pass;
    Executor exec(rt);
    BIS bis(rt, exec, pass);

    // simulate runtime: random touches & access patterns
    std::mt19937_64 rng((unsigned)NowMs());
    std::uniform_int_distribution<int> d(0, (int)all.size()-1);

    // run simulation for some seconds
    for (int t = 0; t < 200; ++t) {
        // randomly touch resources to simulate access patterns (hot tiles)
        int idx = d(rng);
        tele.touch(all[idx],  (size_t)( (double)rt.snapshotCandidates()[idx].residentBytes ));
        // occasionally change importance or risk
        if (t % 25 == 0) {
            tele.setPerceptual(all[ d(rng) ], (double)(rng()%100)/100.0);
        }
        if (t % 40 == 0) {
            // simulate NERVA detection: mark risk randomly
            tele.setRisk(all[d(rng)], (double)(rng()%100)/100.0);
        }
        std::this_thread::sleep_for(50ms);
    }

    // cleanup
    g_log.log(Logger::Info, "Simulation finished. Shutting down...");
    // destructor will join threads
    return 0;

}

