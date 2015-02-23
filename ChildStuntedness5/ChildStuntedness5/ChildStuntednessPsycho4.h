//
//  ChildStuntednessPsycho4.h
//  ChildStuntedness5
//
//  Created by Iaroslav Omelianenko on 2/23/15.
//  Copyright (c) 2015 Nologin. All rights reserved.
//

#ifndef ChildStuntedness5_ChildStuntednessPsycho4_h
#define ChildStuntedness5_ChildStuntednessPsycho4_h

#define LOCAL


#include <sys/time.h>
//#include <emmintrin.h>

const double TRAINING_TIME = 60 * 13;

#ifdef LOCAL
#include "stdc++.h"
int THREADS_NO = 3;
#else
#include <bits/stdc++.h>
int THREADS_NO = 1;
#endif

using namespace std;

#define INLINE   inline __attribute__ ((always_inline))
#define NOINLINE __attribute__ ((noinline))

#define ALIGNED __attribute__ ((aligned(16)))

#define likely(x)   __builtin_expect(!!(x),1)
#define unlikely(x) __builtin_expect(!!(x),0)

#define SSELOAD(a)     _mm_load_si128((__m128i*)&a)
#define SSESTORE(a, b) _mm_store_si128((__m128i*)&a, b)

#define FOR(i,a,b)  for(int i=(a);i<(b);++i)
#define REP(i,a)    FOR(i,0,a)
#define ZERO(m)     memset(m,0,sizeof(m))
#define ALL(x)      x.begin(),x.end()
#define PB          push_back
#define S           size()
#define LL          long long
#define ULL         unsigned long long
#define LD          long double
#define MP          make_pair
#define X           first
#define Y           second
#define VC          vector
#define PII         pair <int, int>
#define VI          VC < int >
#define VVI         VC < VI >
#define VVVI        VC < VVI >
#define VPII        VC < PII >
#define VD          VC < double >
#define VVD         VC < VD >
#define VVVD        VC < VVD >
#define VF          VC < float >
#define VVF         VC < VF >
#define VVVF        VC < VVF >
#define VS          VC < string >
#define VVS         VC < VS >
#define DB(a)       cerr << #a << ": " << (a) << endl;

template<class T> void print(VC < T > v) {cerr << "[";if (v.S) cerr << v[0];FOR(i, 1, v.S) cerr << ", " << v[i];cerr << "]" << endl;}
template<class T> string i2s(T x) {ostringstream o; o << x; return o.str();}
VS splt(string s, char c = ' ') {VS all; int p = 0, np; while (np = s.find(c, p), np >= 0) {all.PB(s.substr(p, np - p)); p = np + 1;} all.PB(s.substr(p)); return all;}

double getTime() {
    timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

struct RNG {
    unsigned int MT[624];
    int index;
    
    RNG(int seed = 1) {
        init(seed);
    }
    
    void init(int seed = 1) {
        MT[0] = seed;
        FOR(i, 1, 624) MT[i] = (1812433253UL * (MT[i-1] ^ (MT[i-1] >> 30)) + i);
        index = 0;
    }
    
    void generate() {
        const unsigned int MULT[] = {0, 2567483615UL};
        REP(i, 227) {
            unsigned int y = (MT[i] & 0x8000000UL) + (MT[i+1] & 0x7FFFFFFFUL);
            MT[i] = MT[i+397] ^ (y >> 1);
            MT[i] ^= MULT[y&1];
        }
        FOR(i, 227, 623) {
            unsigned int y = (MT[i] & 0x8000000UL) + (MT[i+1] & 0x7FFFFFFFUL);
            MT[i] = MT[i-227] ^ (y >> 1);
            MT[i] ^= MULT[y&1];
        }
        unsigned int y = (MT[623] & 0x8000000UL) + (MT[0] & 0x7FFFFFFFUL);
        MT[623] = MT[623-227] ^ (y >> 1);
        MT[623] ^= MULT[y&1];
    }
    
    unsigned int rand() {
        if (index == 0) {
            generate();
        }
        
        unsigned int y = MT[index];
        y ^= y >> 11;
        y ^= y << 7  & 2636928640UL;
        y ^= y << 15 & 4022730752UL;
        y ^= y >> 18;
        index = index == 623 ? 0 : index + 1;
        return y;
    }
    
    INLINE int next() {
        return rand();
    }
    
    INLINE int next(int x) {
        return rand() % x;
    }
    
    INLINE int next(int a, int b) {
        return a + (rand() % (b - a));
    }
    
    INLINE double nextDouble() {
        return (rand() + 0.5) * (1.0 / 4294967296.0);
    }
};

struct TreeNode {
    int level;
    int feature;
    double value;
    double result;
    int left;
    int right;
    
    TreeNode() {
        level = -1;
        feature = -1;
        value = 0;
        result = 0;
        left = -1;
        right = -1;
    }
};

struct RandomForestConfig {
    static const int MSE = 0;
    static const int MCE = 1;
    static const int MAE = 2;
    static const int CUSTOM = 3;
    
    VI randomFeatures = {5};
    VI randomPositions = {2};
    int featuresIgnored = 0;
    int groupFeature = -1;
    VI groups = {};
    int maxLevel = 100;
    int maxNodeSize = 1;
    int maxNodes = 0;
    int threadsNo = 1;
    int treesNo = 1;
    double bagSize = 1.0;
    double timeLimit = 0;
    int lossFunction = MSE;
    bool useBootstrapping = true;
    bool computeImportances = false;
    bool computeOOB = false; //NOT IMPLEMENTED
    
    //Boosting
    bool useLineSearch = false;
    double shrinkage = 0.1;
};

class DecisionTree {
public:
    VC<TreeNode> nodes;
    VD importances;
    
private:
    template <class T> INLINE T customLoss(T x) {
        // const double ALPHA = .25;
        // return abs(x) < ALPHA ? x * x / 2.0 : ALPHA * (abs(x) - ALPHA / 2);
        return abs(x) * sqrt(abs(x));
    }
    
    
public:
    
    DecisionTree() { }
    
    template <class T> INLINE DecisionTree(VC<VC<T>> &features, VC<T> &results, RandomForestConfig &config, int seed, const int USE_WEIGHTS, const int SCORING_FUNCTION) {
        RNG r(seed);
        
        if (config.computeImportances) {
            importances = VD(features[0].S);
        }
        
        VI chosenGroups(features.S);
        if (config.groupFeature == -1 && config.groups.S == 0) {
            REP(i, (int)(features.S * config.bagSize)) chosenGroups[r.next(features.S)]++;
        } else if (config.groupFeature != -1) {
            assert(config.groupFeature >= 0 && config.groupFeature < features.S);
            unordered_map<T, int> groups;
            int groupsNo = 0;
            REP(i, features.S) if (!groups.count(features[i][config.groupFeature])) {
                groups[features[i][config.groupFeature]] = groupsNo++;
            }
            VI groupSize(groupsNo);
            REP(i, (int)(groupsNo * config.bagSize)) groupSize[r.next(groupsNo)]++;
            REP(i, features.S) chosenGroups[i] = groupSize[groups[features[i][config.groupFeature]]];
        } else {
            assert(config.groups.S == features.S);
            int groupsNo = 0;
            for (int x : config.groups) groupsNo = max(groupsNo, x + 1);
            VI groupSize(groupsNo);
            REP(i, (int)(groupsNo * config.bagSize)) groupSize[r.next(groupsNo)]++;
            REP(i, features.S) chosenGroups[i] = groupSize[config.groups[i]];
        }
        
        int bagSize = 0;
        REP(i, features.S) if (chosenGroups[i]) bagSize++;
        
        VI bag(bagSize);
        VI weight(features.S);
        
        int pos = 0;
        
        REP(i, features.S) {
            weight[i] = config.useBootstrapping ? chosenGroups[i] : min(1, chosenGroups[i]);
            if (chosenGroups[i]) bag[pos++] = i;
        }
        
        TreeNode root;
        root.level = 0;
        root.left = 0;
        root.right = pos;
        nodes.PB(root);
        
        queue<int> stack;
        stack.push(0);
        
        while (!stack.empty()) {
            bool equal = true;
            
            int curNode = stack.front(); stack.pop();
            
            int bLeft = nodes[curNode].left;
            int bRight = nodes[curNode].right;
            int bSize = bRight - bLeft;
            
            int totalWeight = 0;
            T totalSum = 0;
            T total2Sum = 0;
            FOR(i, bLeft, bRight) {
                if (USE_WEIGHTS) {
                    totalSum += results[bag[i]] * weight[bag[i]];
                    totalWeight += weight[bag[i]];
                    if (SCORING_FUNCTION == RandomForestConfig::MSE) total2Sum += results[bag[i]] * results[bag[i]] * weight[bag[i]];
                } else {
                    totalSum += results[bag[i]];
                    if (SCORING_FUNCTION == RandomForestConfig::MSE) total2Sum += results[bag[i]] * results[bag[i]];
                }
            }
            assert(bSize > 0);
            
            if (!USE_WEIGHTS) totalWeight = bSize;
            
            FOR(i, bLeft + 1, bRight) if (results[bag[i]] != results[bag[i - 1]]) {
                equal = false;
                break;
            }
            
            if (equal || bSize <= config.maxNodeSize || nodes[curNode].level >= config.maxLevel || config.maxNodes && nodes.S >= config.maxNodes) {
                nodes[curNode].result = totalSum / totalWeight;
                continue;
            }
            
            int bestFeature = -1;
            int bestLeft = 0;
            int bestRight = 0;
            T bestValue = 0;
            T bestLoss = 1e99;
            
            const int randomFeatures = config.randomFeatures[min(nodes[curNode].level, (int)config.randomFeatures.S - 1)];
            REP(i, randomFeatures) {
                
                int featureID = config.featuresIgnored + r.next(features[0].S - config.featuresIgnored);
                
                T vlo, vhi;
                vlo = vhi = features[bag[bLeft]][featureID];
                FOR(j, bLeft + 1, bRight) {
                    vlo = min(vlo, features[bag[j]][featureID]);
                    vhi = max(vhi, features[bag[j]][featureID]);
                }
                if (vlo == vhi) continue;
                
                const int randomPositions = config.randomPositions[min(nodes[curNode].level, (int)config.randomPositions.S - 1)];
                REP(j, randomPositions) {
                    T splitValue = features[bag[bLeft + r.next(bSize)]][featureID];
                    if (splitValue == vlo) {
                        j--;
                        continue;
                    }
                    
                    if (SCORING_FUNCTION == RandomForestConfig::MSE) {
                        T sumLeft = 0;
                        T sum2Left = 0;
                        int totalLeft = 0;
                        FOR(k, bLeft, bRight) {
                            int p = bag[k];
                            if (features[p][featureID] < splitValue) {
                                if (USE_WEIGHTS) {
                                    sumLeft += results[p] * weight[p];
                                    sum2Left += results[p] * results[p] * weight[p];
                                } else {
                                    sumLeft += results[p];
                                    sum2Left += results[p] * results[p];
                                }
                                totalLeft++;
                            }
                        }
                        
                        T sumRight = totalSum - sumLeft;
                        T sum2Right = total2Sum - sum2Left;
                        int totalRight = bSize - totalLeft;
                        
                        if (totalLeft == 0 || totalRight == 0)
                            continue;
                        
                        T loss = sum2Left - sumLeft * sumLeft / totalLeft + sum2Right - sumRight * sumRight / totalRight;
                        
                        if (loss < bestLoss) {
                            bestLoss = loss;
                            bestValue = splitValue;
                            bestFeature = featureID;
                            bestLeft = totalLeft;
                            bestRight = totalRight;
                            if (loss == 0) goto outer;
                        }
                    } else {
                        T sumLeft = 0;
                        int totalLeft = 0;
                        int weightLeft = 0;
                        FOR(k, bLeft, bRight) {
                            int p = bag[k];
                            if (features[p][featureID] < splitValue) {
                                if (USE_WEIGHTS) {
                                    sumLeft += results[p] * weight[p];
                                    weightLeft += weight[p];
                                } else {
                                    sumLeft += results[p];
                                }
                                totalLeft++;
                            }
                        }
                        
                        if (!USE_WEIGHTS) weightLeft = totalLeft;
                        
                        T sumRight = totalSum - sumLeft;
                        int weightRight = totalWeight - weightLeft;
                        int totalRight = bSize - totalLeft;
                        
                        if (totalLeft == 0 || totalRight == 0)
                            continue;
                        
                        T meanLeft = sumLeft / weightLeft;
                        T meanRight = sumRight / weightRight;
                        T loss = 0;
                        
                        if (SCORING_FUNCTION == RandomForestConfig::MCE) {
                            FOR(k, bLeft, bRight) {
                                int p = bag[k];
                                if (features[p][featureID] < splitValue) {
                                    loss += abs(results[p] - meanLeft)  * (results[p] - meanLeft)  * (results[p] - meanLeft)  * weight[p];
                                } else {
                                    loss += abs(results[p] - meanRight) * (results[p] - meanRight) * (results[p] - meanRight) * weight[p];
                                }
                                if (loss > bestLoss) break; //OPTIONAL
                            }
                        } else if (SCORING_FUNCTION == RandomForestConfig::MAE) {
                            FOR(k, bLeft, bRight) {
                                int p = bag[k];
                                if (features[p][featureID] < splitValue) {
                                    loss += abs(results[p] - meanLeft)  * weight[p];
                                } else {
                                    loss += abs(results[p] - meanRight) * weight[p];
                                }
                                if (loss > bestLoss) break; //OPTIONAL
                            }
                        } else if (SCORING_FUNCTION == RandomForestConfig::CUSTOM) {
                            FOR(k, bLeft, bRight) {
                                int p = bag[k];
                                if (features[p][featureID] < splitValue) {
                                    loss += customLoss(results[p] - meanLeft)  * weight[p];
                                } else {
                                    loss += customLoss(results[p] - meanRight) * weight[p];
                                }
                                if (loss > bestLoss) break; //OPTIONAL
                            }
                        }
                        
                        if (loss < bestLoss) {
                            bestLoss = loss;
                            bestValue = splitValue;
                            bestFeature = featureID;
                            bestLeft = totalLeft;
                            bestRight = totalRight;
                            if (loss == 0) goto outer;
                        }
                    }
                }
            }
        outer:
            
            if (bestLeft == 0 || bestRight == 0) {
                nodes[curNode].result = totalSum / totalWeight;
                continue;
            }
            
            if (config.computeImportances) {
                importances[bestFeature] += bRight - bLeft;
            }
            
            T mean = totalSum / totalWeight;
            
            T nextValue = -1e99;
            FOR(i, bLeft, bRight) if (features[bag[i]][bestFeature] < bestValue) nextValue = max(nextValue, features[bag[i]][bestFeature]);
            
            TreeNode left;
            TreeNode right;
            
            left.level = right.level = nodes[curNode].level + 1;
            nodes[curNode].feature = bestFeature;
            nodes[curNode].value = (bestValue + nextValue) / 2.0;
            if (!(nodes[curNode].value > nextValue)) nodes[curNode].value = bestValue;
            nodes[curNode].left = nodes.S;
            nodes[curNode].right = nodes.S + 1;
            
            int bMiddle = bRight;
            FOR(i, bLeft, bMiddle) {
                if (features[bag[i]][nodes[curNode].feature] >= nodes[curNode].value) {
                    swap(bag[i], bag[--bMiddle]);
                    i--;
                    continue;
                }
            }
            
            assert(bestLeft == bMiddle - bLeft);
            assert(bestRight == bRight - bMiddle);
            
            left.left = bLeft;
            left.right = bMiddle;
            right.left = bMiddle;
            right.right = bRight;
            
            // stack.PB(nodes.S);
            // stack.PB(nodes.S + 1);
            stack.push(nodes.S);
            stack.push(nodes.S + 1);
            
            nodes.PB(left);
            nodes.PB(right);
            
        }
        
        nodes.shrink_to_fit();
    }
    
    template <class T> double predict(VC<T> &features) {
        TreeNode *pNode = &nodes[0];
        while (true) {
            if (pNode->feature < 0) return pNode->result;
            pNode = &nodes[features[pNode->feature] < pNode->value ? pNode->left : pNode->right];
        }
    }
};

RNG gRNG(1);

class TreeEnsemble {
public:
    
    VC<DecisionTree> trees;
    VD importances;
    RandomForestConfig config;
    
    void clear() {
        trees.clear();
        trees.shrink_to_fit();
    }
    
    template <class T> DecisionTree createTree(VC<VC<T>> &features, VC<T> &results, RandomForestConfig &config, int seed) {
        if (config.useBootstrapping) {
            if (config.lossFunction == RandomForestConfig::MAE) {
                return DecisionTree(features, results, config, seed, true, RandomForestConfig::MAE);
            } else if (config.lossFunction == RandomForestConfig::MSE) {
                return DecisionTree(features, results, config, seed, true, RandomForestConfig::MSE);
            } else if (config.lossFunction == RandomForestConfig::MCE) {
                return DecisionTree(features, results, config, seed, true, RandomForestConfig::MCE);
            } else if (config.lossFunction == RandomForestConfig::CUSTOM) {
                return DecisionTree(features, results, config, seed, true, RandomForestConfig::CUSTOM);
            }
        } else {
            if (config.lossFunction == RandomForestConfig::MAE) {
                return DecisionTree(features, results, config, seed, false, RandomForestConfig::MAE);
            } else if (config.lossFunction == RandomForestConfig::MSE) {
                return DecisionTree(features, results, config, seed, false, RandomForestConfig::MSE);
            } else if (config.lossFunction == RandomForestConfig::MCE) {
                return DecisionTree(features, results, config, seed, false, RandomForestConfig::MCE);
            } else if (config.lossFunction == RandomForestConfig::CUSTOM) {
                return DecisionTree(features, results, config, seed, false, RandomForestConfig::CUSTOM);
            }
        }
    }
    
    LL countTotalNodes() {
        LL rv = 0;
        REP(i, trees.S) rv += trees[i].nodes.S;
        return rv;
    }
    
    void printImportances() {
        assert(config.computeImportances);
        
        VC<pair<double, int>> vp;
        REP(i, importances.S) vp.PB(MP(importances[i], i));
        sort(vp.rbegin(), vp.rend());
        
        REP(i, importances.S) printf("#%02d: %.6lf\n", vp[i].Y, vp[i].X);
    }
    
};

class RandomForest : public TreeEnsemble {
public:
    
    template <class T> void train(VC<VC<T>> &features, VC<T> &results, RandomForestConfig &_config, int treesMultiplier = 1) {
        double startTime = getTime();
        config = _config;
        
        int treesNo = treesMultiplier * config.treesNo;
        
        if (config.threadsNo == 1) {
            REP(i, treesNo) {
                if (config.timeLimit && getTime() - startTime > config.timeLimit) break;
                trees.PB(createTree(features, results, config, gRNG.next()));
            }
        } else {
            thread *threads = new thread[config.threadsNo];
            mutex mutex;
            REP(i, config.threadsNo)
            threads[i] = thread([&] {
                while (true) {
                    mutex.lock();
                    int seed = gRNG.next();
                    mutex.unlock();
                    auto tree = createTree(features, results, config, seed);
                    mutex.lock();
                    if (trees.S < treesNo)
                        trees.PB(tree);
                    bool done = trees.S >= treesNo || config.timeLimit && getTime() - startTime > config.timeLimit;
                    mutex.unlock();
                    if (done) break;
                }
            });
            REP(i, config.threadsNo) threads[i].join();
            delete[] threads;
        }
        
        if (config.computeImportances) {
            importances = VD(features[0].S);
            for (DecisionTree tree : trees)
                REP(i, importances.S)
                importances[i] += tree.importances[i];
            double sum = 0;
            REP(i, importances.S) sum += importances[i];
            REP(i, importances.S) importances[i] /= sum;
        }
    }
    
    template <class T> double predict(VC<T> &features) {
        assert(trees.S);
        
        double sum = 0;
        REP(i, trees.S) sum += trees[i].predict(features);
        return sum / trees.S;
    }
    
    template <class T> VD predict(VC<VC<T>> &features) {
        assert(trees.S);
        
        int samplesNo = features.S;
        
        VD rv(samplesNo);
        if (config.threadsNo == 1) {
            REP(j, samplesNo) {
                REP(i, trees.S) rv[j] += trees[i].predict(features[j]);
                rv[j] /= trees.S;
            }
        } else {
            thread *threads = new thread[config.threadsNo];
            REP(i, config.threadsNo)
            threads[i] = thread([&](int offset) {
                for (int j = offset; j < samplesNo; j += config.threadsNo) {
                    REP(k, trees.S) rv[j] += trees[k].predict(features[j]);
                    rv[j] /= trees.S;
                }
            }, i);
            REP(i, config.threadsNo) threads[i].join();
            delete[] threads;
        }
        return rv;
    }
    
    // template < class T > void addSample(VC < T > &features, T result) {
    // REP(i, trees.S) {
    // int weight = gRNG.poisson(1);
    // if (weight > 0)	trees[i].addSample(features, result, weight);
    // }
    // }
    
};

class BoostedForest : public TreeEnsemble {
public:
    
    VD currentResults;
    
    void clear() {
        trees.clear();
        trees.shrink_to_fit();
        currentResults.clear();
    }
    
    template <class T> void train(VC<VC<T>> &features, VC<T> &results, RandomForestConfig &_config, int treesMultiplier = 1) {
        double startTime = getTime();
        config = _config;
        
        int treesNo = treesMultiplier * config.treesNo;
        
        if (currentResults.S == 0) currentResults = VD(results.S);
        
        if (config.threadsNo == 1) {
            VC<T> residuals(results.S);
            REP(i, treesNo) {
                if (config.timeLimit && getTime() - startTime > config.timeLimit) break;
                REP(j, results.S) residuals[j] = results[j] - currentResults[j];
                trees.PB(createTree(features, residuals, config, gRNG.next()));
                REP(j, results.S) currentResults[j] += trees[trees.S-1].predict(features[j]) * config.shrinkage;
            }
        } else {
            //TODO: improve MT speed
            mutex mutex;
            for (int i = 0; i < treesNo; i += config.threadsNo) {
                if (config.timeLimit && getTime() - startTime > config.timeLimit) break;
                
                int usedThreads = min(config.threadsNo, treesNo - i);
                VC<T> residuals(results.S);
                REP(j, results.S) residuals[j] = results[j] - currentResults[j];
                
                thread *threads = new thread[config.threadsNo];
                REP(j, usedThreads)
                threads[j] = thread([&] {
                    mutex.lock();
                    int seed = gRNG.next();
                    mutex.unlock();
                    
                    auto tree = createTree(features, residuals, config, seed);
                    VD estimates(results.S);
                    REP(k, estimates.S) estimates[k] = tree.predict(features[k]) * config.shrinkage;
                    
                    mutex.lock();
                    trees.PB(tree);
                    REP(k, estimates.S) currentResults[k] += estimates[k];
                    mutex.unlock();
                });
                
                REP(j, usedThreads) threads[j].join();
                delete[] threads;
            }
        }
        
        if (config.computeImportances) {
            importances = VD(features[0].S);
            for (DecisionTree tree : trees)
                REP(i, importances.S)
                importances[i] += tree.importances[i];
            double sum = 0;
            REP(i, importances.S) sum += importances[i];
            REP(i, importances.S) importances[i] /= sum;
        }
    }
    
    template <class T> double predict(VC<T> &features) {
        assert(trees.S);
        
        double sum = 0;
        if (config.threadsNo == 1) {
            REP(i, trees.S) sum += trees[i].predict(features);
        } else {
            thread *threads = new thread[config.threadsNo];
            VD sums(config.threadsNo);
            int order = 0;
            REP(i, config.threadsNo)
            threads[i] = thread([&](int offset) {
                for (int j = offset; j < trees.S; j += config.threadsNo)
                    sums[offset] += trees[j].predict(features);
            }, i);
            REP(i, config.threadsNo) threads[i].join();
            REP(i, config.threadsNo) sum += sums[i];
            delete[] threads;
        }
        return sum * config.shrinkage;
    }
};

static RNG rng;


struct Sample {
    static const int UNKNOWN = -1000;
    static const int RANGE = 6;
    
    int id;
    double agedays;
    double muaccm;
    double sftmm;
    double muaz;
    double sftz;
    double bfed;
    double wean;
    double siteid;
    int sex;
    double gagebrth;
    double brthweek;
    double birthwg;
    double birthlen;
    double birthhc;
    double mage;
    double mhtcm;
    double fhtcm;
    double mparity;
    double wtkg;
    double lencm;
    double hcircm;
    
    static double convert(string s) {
        return s == "." ? UNKNOWN : atof(s.c_str());
    }
    
    static double convertx(double s) {
        return s == UNKNOWN ? 0.0 : s;
    }
    
    template<class T> void correctValue(T &v, bool validUnknown = true) {
        if (v == UNKNOWN) {
            if (!validUnknown) v = 1;
        } else {
            if (v <= -RANGE || v >= RANGE) v = 0;
        }
    }
    
    Sample() { }
    
    Sample(string s) {
        VS v = splt(s, ',');
        assert(v.S == 22);
        
        int pos = 0;
        
        id = convert(v[pos++]);
        agedays = max(0.0, min(900.0, convert(v[pos++])));
        muaccm = convert(v[pos++]);
        sftmm = convert(v[pos++]);
        muaz = convert(v[pos++]);
        sftz = convert(v[pos++]);
        bfed = convert(v[pos++]);
        wean = convert(v[pos++]);
        siteid = convert(v[pos++]);
        sex = convert(v[pos++]);
        gagebrth = convert(v[pos++]);
        brthweek = convert(v[pos++]);
        birthwg = convert(v[pos++]);
        birthlen = convert(v[pos++]);
        birthhc = convert(v[pos++]);
        mage = convert(v[pos++]);
        mhtcm = convert(v[pos++]);
        fhtcm = convert(v[pos++]);
        mparity = convert(v[pos++]);
        wtkg = convert(v[pos++]);
        lencm = convert(v[pos++]);
        hcircm = convert(v[pos++]);
        
        assert(pos == v.S);
        
        
        if (sex != 1 && sex != 2) sex = 1;
        //TODO: add other corrections
    }
};


struct ExpValue {
    static const int BAGS = 111;
    static constexpr double MAX_TIME = 1000;
    double bags[BAGS];
    
    int calculateBag(double t) {
        return (int)(t * BAGS / MAX_TIME);
    }
    
    void train(VF &t0, VF &t1, VF &v0, VF &v1) {
        ZERO(bags);
        VI bagsNo(BAGS);
        REP(i, t0.S) {
            REP(j, BAGS) {
                double t = j * MAX_TIME / BAGS;
                if (t < t0[i] || t > t1[i]) continue;
                double w0 = (t - t0[i]) / (t1[i] - t0[i]);
                bags[j] += v0[i] * (1 - w0) + v1[i] * w0;
                bagsNo[j]++;
            }
        }
        
        REP(i, BAGS) if (bagsNo[i]) bags[i] /= bagsNo[i];
        
        for (int i = 1; i < BAGS; i++)      if (bagsNo[i] == 0) bags[i] = bags[i - 1];
        for (int i = BAGS - 2; i >= 0; i--) if (bagsNo[i] == 0) bags[i] = bags[i + 1];
    }
    
    double predict(double t) {
        int bag = calculateBag(t);
        double w = (t - bag * MAX_TIME / BAGS) / (MAX_TIME / BAGS);
        return bags[bag] * (1 - w) + bags[bag+1] * w;
    }
    
    double predict(Sample &s) {
        return predict(s.agedays);
    }
};

ExpValue expValue[3][5];


VD extractFeatures(VC<Sample> &samples, int pos, int type = 0) {
    int sex = samples[0].sex == 1;
    
    VD rv;
    rv.PB(samples[pos].id);
    rv.PB(sex);
    rv.PB(samples[pos].siteid);
    rv.PB(samples[pos].agedays);
    
    int lp = pos - 1, hp = pos + 1;
    while (lp >= 0        && samples[lp].muaccm == Sample::UNKNOWN) lp--;
    while (hp < samples.S && samples[hp].muaccm == Sample::UNKNOWN) hp++;
    if (type != 2) {
        rv.PB(lp == -1 ? Sample::UNKNOWN : samples[lp].muaccm);
        rv.PB(lp == -1 ? Sample::UNKNOWN : samples[lp].sftmm);
        rv.PB(hp == samples.S ? Sample::UNKNOWN : samples[hp].muaccm);
        rv.PB(hp == samples.S ? Sample::UNKNOWN : samples[hp].sftmm);
    }
    
    if (samples[pos].muaccm != Sample::UNKNOWN) {
        rv.PB(samples[pos].muaccm);
        rv.PB(samples[pos].sftmm);
        rv.PB(samples[pos].muaccm - expValue[sex][3].predict(samples[pos]));
        rv.PB(samples[pos].sftmm  - expValue[sex][4].predict(samples[pos]));
        rv.PB(samples[pos].muaz);
        rv.PB(samples[pos].sftz);
    } else if (lp >= 0 && hp < samples.S) {
        double w = 1.0 * (samples[pos].agedays - samples[lp].agedays) / (samples[hp].agedays - samples[lp].agedays);
        rv.PB(samples[lp].muaccm * (1 - w) + samples[hp].muaccm * w);
        rv.PB(samples[lp].sftmm  * (1 - w) + samples[hp].sftmm  * w);
        rv.PB(samples[lp].muaccm * (1 - w) + samples[hp].muaccm * w - expValue[sex][3].predict(samples[pos]));
        rv.PB(samples[lp].sftmm  * (1 - w) + samples[hp].sftmm  * w - expValue[sex][4].predict(samples[pos]));
        rv.PB(samples[lp].muaz * (1 - w) + samples[hp].muaz * w);
        rv.PB(samples[lp].sftz * (1 - w) + samples[hp].sftz * w);
    } else {
        rv.PB(Sample::UNKNOWN);
        rv.PB(Sample::UNKNOWN);
        rv.PB(Sample::UNKNOWN);
        rv.PB(Sample::UNKNOWN);
        rv.PB(Sample::UNKNOWN);
        rv.PB(Sample::UNKNOWN);
    }
    
    rv.PB(samples[pos].mage);
    rv.PB(samples[pos].mparity);
    rv.PB(samples[pos].mhtcm);
    rv.PB(samples[pos].fhtcm);
    
    rv.PB(samples[pos].gagebrth);
    rv.PB(samples[pos].birthwg);
    rv.PB(samples[pos].birthlen);
    rv.PB(samples[pos].birthhc);
    
    VI no(5);
    VD sums2(5);
    VD maxv(5, -1e6);
    VD minv(5, +1e6);
    VVD values(5);
    VVD allvalues(5);
    for (int i = samples.S - 1; i >= 0; i--) {
        double v[] = {samples[i].wtkg, samples[i].lencm, samples[i].hcircm, samples[i].muaccm, samples[i].sftmm};
        REP(j, sizeof(v)/sizeof(double)) if (v[j] != Sample::UNKNOWN) {
            double x = v[j] - expValue[sex][j].predict(samples[i]);
            allvalues[j].PB(x);
            if (i >= pos) continue;
            no[j]++;
            maxv[j] = max(maxv[j], x);
            minv[j] = min(minv[j], x);
            values[j].PB(x);
            if (no[j] <= 3) sums2[j] += x;
        }
    }
    rv.PB(pos);
    
    REP(i, 5) {
        if (no[i] == 0) {
            rv.PB(Sample::UNKNOWN);
        } else {
            sort(ALL(values[i]));
            rv.PB(values[i].S % 2 ? values[i][values[i].S / 2] : (values[i][values[i].S / 2 - 1] + values[i][values[i].S / 2]) / 2);
        }
        
        if (type == 2 && i < 2) {
            if (no[i] == 0) {
                rv.PB(Sample::UNKNOWN);
            } else {
                sort(ALL(allvalues[i]));
                rv.PB(allvalues[i].S % 2 ? allvalues[i][allvalues[i].S / 2] : (allvalues[i][allvalues[i].S / 2 - 1] + allvalues[i][allvalues[i].S / 2]) / 2);
            }
        }
        
        if (i < 3) {
            if (type != 2) rv.PB(no[i] == 0 ? Sample::UNKNOWN : sums2[i] / min(3, no[i]));
            rv.PB(no[i] == 0 ? Sample::UNKNOWN : maxv[i]);
            rv.PB(no[i] == 0 ? Sample::UNKNOWN : minv[i]);
        }
    }
    
    if (type == 2) {
        rv.PB(samples[pos].wtkg - expValue[sex][0].predict(samples[pos]));
        rv.PB(samples[pos].lencm - expValue[sex][1].predict(samples[pos]));
        rv.PB(pos ? samples[pos-1].hcircm - expValue[sex][2].predict(samples[pos-1]) : Sample::UNKNOWN);
    } 	
    return rv;
}

VC<VC<Sample>> splitSamples(VS &data) {
    VC<VC<Sample>> rv;
    REP(i, data.S) {
        int sp = i;
        int ep = i;
        int curId = Sample(data[sp]).id;
        VC<Sample> samples;
        while (ep < data.S && Sample(data[ep]).id == curId) {
            samples.PB(Sample(data[ep]));
            ep++;
        }
        rv.PB(samples);
        i = ep - 1;
    }
    return rv;
}

void extractExpectedValues(VC<VC<Sample>> &data) {
    VVVF vt0(3, VVF(5, VF()));
    VVVF vt1(3, VVF(5, VF()));
    VVVF vv0(3, VVF(5, VF()));
    VVVF vv1(3, VVF(5, VF()));
    
    for (VC<Sample> samples : data) {
        REP(i, samples.S - 1) {
            REP(csex, 3) {
                int sex = samples[i].sex == 1;
                if (sex != csex && csex != 2) continue;
                
                double v0[] = {samples[i].wtkg, samples[i].lencm, samples[i].hcircm, samples[i].muaccm, samples[i].sftmm};
                double v1[] = {samples[i+1].wtkg, samples[i+1].lencm, samples[i+1].hcircm, samples[i+1].muaccm, samples[i+1].sftmm};
                double t0 = samples[i].agedays;
                double t1 = samples[i+1].agedays;
                REP(j, 5) if (v0[j] != Sample::UNKNOWN && v1[j] != Sample::UNKNOWN) {
                    vt0[csex][j].PB(t0);
                    vt1[csex][j].PB(t1);
                    vv0[csex][j].PB(v0[j]);
                    vv1[csex][j].PB(v1[j]);
                }
            }
        }
    }
    
    REP(i, 3) REP(j, 5) expValue[i][j].train(vt0[i][j], vt1[i][j], vv0[i][j], vv1[i][j]);
}

void correctData(VC<VC<Sample>> &data) {
    for (VC<Sample> &samples : data) {
        REP(j, samples.S) {
            if (samples[j].muaccm == Sample::UNKNOWN && samples[j].sftmm == Sample::UNKNOWN) {
                int lp = j - 1, hp = j + 1;
                while (lp >= 0        && samples[lp].muaccm == Sample::UNKNOWN) lp--;
                while (hp < samples.S && samples[hp].muaccm == Sample::UNKNOWN) hp++;
                if (lp >= 0 && hp < samples.S) {
                    double w = 1.0 * (samples[j].agedays - samples[lp].agedays) / (samples[hp].agedays - samples[lp].agedays);
                    samples[j].muaccm = samples[lp].muaccm * (1 - w) + samples[hp].muaccm * w;
                    samples[j].sftmm  = samples[lp].sftmm  * (1 - w) + samples[hp].sftmm  * w;
                }
            }
        }
    }
    
}

class ChildStuntedness4 {
    
    VVVD models;
    
    VVD blends = {{0, 1, 1}, {0, 1, 1}, {0, 1, 1}};
    
public: VS predict(int testType, VS &training, VS &testing) {
    double startTime = getTime();
    
    const int TREES_NO = 6666;
    
    RandomForestConfig cfg;
    cfg.featuresIgnored = 0;
    cfg.useBootstrapping = false;
    cfg.bagSize = 0.5;
    cfg.shrinkage = 0.005;
    cfg.maxLevel = 7;
    cfg.randomFeatures = {2, 3, 4, 5, 6};
    cfg.randomPositions = {1};
    cfg.threadsNo = THREADS_NO;
    cfg.lossFunction = RandomForestConfig::MAE;
    
    VVD trainDataW;
    VVD trainDataL;
    VVD trainDataH;
    VD resultsW;
    VD resultsL;
    VD resultsH;
    
    REP(i, testing.S) training.PB(testing[i]);
    
    VC<VC<Sample>> trainSamples = splitSamples(training);
    VC<VC<Sample>> testSamples = splitSamples(testing);
    
    extractExpectedValues(trainSamples);
    
    correctData(trainSamples);
    correctData(testSamples);
    
    BoostedForest BFW;
    BoostedForest BFL;
    BoostedForest BFH;
    
    for (VC<Sample> samples : trainSamples) {
        REP(j, samples.S) {
            if (samples[j].wtkg == Sample::UNKNOWN || samples[j].lencm == Sample::UNKNOWN) continue;
            
            int sex = samples[j].sex == 1;
            
            VD featuresH = extractFeatures(samples, j, 2);
            if (samples[j].hcircm != Sample::UNKNOWN) {
                trainDataH.PB(featuresH);
                resultsH.PB(samples[j].hcircm - expValue[sex][2].predict(samples[j]));
            } 
        }
    }
    
    BFH.train(trainDataH, resultsH, cfg, TREES_NO);
    trainDataH.clear();
    resultsH.clear();
    
    for (VC<Sample> samples : trainSamples) {
        REP(j, samples.S) {
            if (samples[j].wtkg == Sample::UNKNOWN || samples[j].lencm == Sample::UNKNOWN) continue;
            
            int sex = samples[j].sex == 1;
            
            VD featuresW = extractFeatures(samples, j, 0);
            if (samples[j].wtkg != Sample::UNKNOWN) {
                trainDataW.PB(featuresW);
                resultsW.PB(samples[j].wtkg - expValue[sex][0].predict(samples[j]));
            }
            
            VD featuresL = extractFeatures(samples, j, 1);
            if (samples[j].lencm != Sample::UNKNOWN) {
                trainDataL.PB(featuresL);
                resultsL.PB(samples[j].lencm - expValue[sex][1].predict(samples[j]));
            }
            
            VD featuresH = extractFeatures(samples, j, 2);
            if (samples[j].hcircm != Sample::UNKNOWN) {
                trainDataH.PB(featuresH);
                resultsH.PB(samples[j].hcircm - expValue[sex][2].predict(samples[j]));
            } else {
                samples[j].hcircm = BFH.predict(featuresH) + expValue[sex][2].predict(samples[j]);
            }
        }
    }
    
    BFH.clear();
    // cfg.lossFunction = RandomForestConfig::MSE;
    // BFW.train(trainDataW, resultsW, cfg, TREES_NO);
    // cfg.lossFunction = RandomForestConfig::MAE;
    // BFL.train(trainDataL, resultsL, cfg, TREES_NO);
    // BFH.train(trainDataH, resultsH, cfg, TREES_NO);
    
    int treesNo = 0;
    while (getTime() - startTime < TRAINING_TIME) {
        cfg.lossFunction = RandomForestConfig::MSE;
        BFW.train(trainDataW, resultsW, cfg, 1);
        cfg.lossFunction = RandomForestConfig::MAE;
        BFL.train(trainDataL, resultsL, cfg, 1);
        BFH.train(trainDataH, resultsH, cfg, 1);
        treesNo++;
        if (treesNo >= TREES_NO) break;
    }
    DB(treesNo);
    
    VS rv;
    for (VC<Sample> samples : testSamples) {
        REP(j, samples.S) {
            Sample &s = samples[j];
            int sex = samples[j].sex == 1;
            
            double predW = s.wtkg;
            double predL = s.lencm;
            double predH = s.hcircm;
            
            double pW = Sample::UNKNOWN;
            double pL = Sample::UNKNOWN;
            double pH = Sample::UNKNOWN;
            
            VD featuresW = extractFeatures(samples, j, 0);
            if (predW == Sample::UNKNOWN) {
                pW = BFW.predict(featuresW);
                samples[j].wtkg = pW + expValue[sex][0].predict(s);
                predW = pW * blends[0][2] + expValue[sex][0].predict(s) * blends[0][1] + blends[0][0];
            } 
            
            VD featuresL = extractFeatures(samples, j, 1);
            if (predL == Sample::UNKNOWN) {
                pL = BFL.predict(featuresL);
                samples[j].lencm = pL + expValue[sex][1].predict(s);
                predL = pL * blends[1][2] + expValue[sex][1].predict(s) * blends[1][1] + blends[1][0];
            } 
            
            VD featuresH = extractFeatures(samples, j, 2);
            if (predH == Sample::UNKNOWN) {
                pH = BFH.predict(featuresH);
                samples[j].hcircm = pH + expValue[sex][2].predict(s);
                predH = pH * blends[2][2] + expValue[sex][2].predict(s) * blends[2][1] + blends[2][0];
            } 
            
            rv.PB(i2s(predW) + "," + i2s(predL) + "," + i2s(predH));
        }
    }
    
    return rv;
}
    
    void saveModel(string fn, VVD &model) {
        FILE *f = fopen(fn.c_str(), "a");
        for (VD &v : model) {
            for (double d : v) fprintf(f, "%.10lf ", d);
            fprintf(f, "\n");
        }
        fclose(f);
    }
    
    void saveModels(VVD &correct) {
        saveModel("modelx.txt", correct);
        REP(i, models.S) saveModel("model" + i2s(i) + ".txt", models[i]);
    }
    
};


#endif
