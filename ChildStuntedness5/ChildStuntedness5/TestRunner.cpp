//
//  TestRunner.cpp
//  TripSafety
//
//  Created by Iaroslav Omelianenko on 1/16/15.
//  Copyright (c) 2015 yaric. All rights reserved.
//
#include <iostream>

#define FROM_TEST_RUNNER

#include "stdc++.h"
#include "ChildStuntedness5.h"

namespace am {
    namespace launcher {
        
        /*
         * The random class generator.
         */
        struct SecureRandom {
            int seed;
            
            SecureRandom(int seed) : seed(seed){}
            
            int nextInt(int max) {
                std::default_random_engine engine(seed);
                std::uniform_int_distribution<int> distribution(0, max - 1);
                
                return distribution(engine);
            }
        };
        
        SecureRandom rnd(1);
        
        VS splt(std::string s, char c = ',') {
            VS all;
            int p = 0, np;
            while (np = (int)s.find(c, p), np >= 0) {
                if (np != p)
                    all.push_back(s.substr(p, np - p));
                else
                    all.push_back("");
                
                p = np + 1;
            }
            if (p < s.size())
                all.push_back(s.substr(p));
            return all;
        }
        
        class ChildStuntedness5Vis {
            int subsetsNum = 9;
            int iqCol = 27 - 1;
            
            VS DTrain;
            VS DTest;
            VI groundTruth;
            double sse0;
            
            VVS dataSamples;
            
            ChildStuntedness5 *test;
            
        public:
            std::string dataFile = "../data/exampleData.csv";
            
            ChildStuntedness5Vis(ChildStuntedness5 *tsf) : test(tsf) {}
            
            double doExec() {
                
                if (!loadTestData()) {
                    std::cerr << "Failed to load test data" << std::endl;
                    return -1;
                }
                
                //
                // Start solution testing
                //
                double score = 0, sse;
                int scenario = 0;
                for (int i = 0; i < subsetsNum; i++) {
                    scenario = i % 3;
                    generateTestData(scenario, i);
                    VD res = test->predict(0, scenario, DTrain, DTest);
                    
                    Assert(groundTruth.size() == res.size(), "Expected results count not equal to found. Expected: %lu, but found: %lu", groundTruth.size(), res.size());
                    sse = 0;
                    for (int j = 0; j < res.size(); j++) {
                        double e = res[j] - groundTruth[j];
                        sse += e * e;
                    }
                    // calculate score
                    double s = 1000000 * fmax(0, 1.0 - sse/sse0);
                    Printf("%i.) Score = %f, sse: %f, sse0: %f\n", i, s, sse, sse0);
                    score += s;
                }
                return score / subsetsNum;
            }
            
        private:
            
            void generateTestData(const int scenario, const int subset) {
                size_t subset_total = dataSamples.size() / subsetsNum;
                int train_start_index = subset * (int)subset_total;
                int test_start_index = train_start_index + subset_total * 0.66;
                int end_index = train_start_index + (int)subset_total;
                if (end_index > dataSamples.size()) {
                    end_index = (int)dataSamples.size();
                }
                
                //
                // prepare train data
                //
                DTrain.clear();
                double mean = 0;
                int countIq = 0;
                for (int i = train_start_index; i < test_start_index; i++) {
                    VS rows = dataSamples[i];
                    for (int r = 0; r < rows.size(); r++) {
                        string line = rows[r];
                        VS values = splt(line);
                        double iq = atof(values[iqCol].c_str());
                        
                        if (scenario > 0) {
                            DTrain.push_back(line);
                        } else if (iq > 0) {
                            // add only line with IQ set for 1 scenario
                            DTrain.push_back(line);
                        }
                        
                        if (iq > 0) {
                            mean += iq;
                            countIq++;
                        }
                    }
                }
//                filterDataSet(DTrain, scenario, true);
                
                //
                // prepare test data
                //
                DTest.clear();
                groundTruth.clear();
                for (int i = test_start_index; i < end_index; i++) {
                    VS rows = dataSamples[i];
                    for (int r = 0; r < rows.size(); r++) {
                        string line = rows[r];
                        VS values = splt(line);
                        double iq = atof(values[iqCol].c_str());
                        if (iq > 0) {
                            groundTruth.push_back(iq);
                            // add only line with IQ set for 1 scenario
                            DTest.push_back(line);
                        } else if (scenario > 0) {
                            DTest.push_back(line);
                        }
                    }
                }
//                filterDataSet(DTest, scenario, false);
                
                //
                // calculate sse0
                //
                mean /= countIq;
                sse0 = 0;
                for (const double &iq : groundTruth) {
                    double e = mean - iq;
                    sse0 += e * e;
                }
            }
            
            void filterDataSet(VS &data, const int scenario, const bool includeClass) {
                VI indices;
                if (scenario == 2) {
                    VI sc3 = {11, 13, 19, 20, 21, 22, 23, 24, 25, 26};
                    indices.insert(indices.begin(), sc3.begin(), sc3.end());
                }
                
                if (scenario >= 1) {
                    VI sc2 = {2, 3, 4, 5, 6, 7, 8, 9, 10};
                    indices.insert(indices.begin(), sc2.begin(), sc2.end());
                }
                
                VI sc1 = {1, 12, 14, 15, 16, 17, 18};
                indices.insert(indices.begin(), sc1.begin(), sc1.end());
                
                for (int i = 0; i < data.size(); i++) {
                    string line = data[i];
                    VS values = splt(line);
                    string newLine;
                    for (int j = 0; j < indices.size(); j++) {
                        newLine.append(values[indices[j] - 1]);
                        if (j < indices.size() - 1) {
                            newLine.append(",");
                        }
                    }
                    if (includeClass) {
                        newLine.append(",");
                        newLine.append(values[iqCol]);
                    }
                    
                    // store new data
                    data[i] = newLine;
                }
            }
            
            bool loadTestData() {
                fprintf(stderr, "Data file: %s\n", dataFile.c_str());
                //
                // load data
                //
                std::ifstream datafile (dataFile);
                if (!datafile.is_open()) {
                    std::cerr << "Error in opening file: " << dataFile << std::endl;
                    return false;
                }

                std::string line;
                int lastIndex = -1, index = -1, row = -1;
                while (! datafile.eof() ) {
                    getline(datafile, line);
                    row++;
                    
                    if (!line.empty()) {
                        // remove \r charcter
                        size_t car_ret_index = line.rfind("\r");
                        line.erase(car_ret_index);
                        
                        VS s = splt(line);
                        
                        int subjid = atof(s[0].c_str());
                        if (subjid - 1 != lastIndex) {
                            VS subjLines;
                            dataSamples.push_back(subjLines);
                            lastIndex = subjid - 1;
                            index ++;
                        }
                        // add line
                        dataSamples[index].push_back(line);
                    } else {
                        Printf("Empty line found at the input data at row: %i!\n", row);
                    }
                    
                }
                datafile.close();
                
                fprintf(stderr, "Loaded: %lu subjects\n", dataSamples.size());
                
                return true;
            }
        };
    }
}

int main(int argc, const char * argv[]) {
    if (argc < 1) {
        printf("Usage: dataFile\n");
        return 0;
    }
    ChildStuntedness5 task;
    am::launcher::ChildStuntedness5Vis runner(&task);
    runner.dataFile = argv[1];
    
    double score = runner.doExec();
    fprintf(stderr, "Score = %f\n", score);
    
    return 0;
}

