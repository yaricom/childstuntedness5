//
//  ChildStuntedness5.h
//  ChildStuntedness5
//
//  Created by Iaroslav Omelianenko on 2/23/15.
//  Copyright (c) 2015 Nologin. All rights reserved.
//

#ifndef ChildStuntedness5_ChildStuntedness5_h
#define ChildStuntedness5_ChildStuntedness5_h

#define LOCAL true

//#define USE_ESTIMATORS
#define USE_REGERESSION

#ifdef LOCAL
#include "stdc++.h"
#else
#include <bits/stdc++.h>
#endif

#include <iostream>
#include <sys/time.h>

using namespace std;

#define FOR(i,a,b)  for(int i=(a);i<(b);++i)
#define LL          long long
#define ULL         unsigned long long
#define LD          long double
#define MP          make_pair
#define VC          vector
#define PII         pair <int, int>
#define VI          VC < int >
#define VVI         VC < VI >
#define VVVI        VC < VVI >
#define VPII        VC < PII >
#define VD          VC < double >
#define VVD         VC < VD >
#define VF          VC < float >
#define VVF         VC < VF >
#define VS          VC < string >
#define VVS         VC < VS >
#define VE          VC <Entry>
#define VVE         VC< VC <Entry> >

template<class T> void print(VC < T > v) {cerr << "[";if (v.size()) cerr << v[0];FOR(i, 1, v.size()) cerr << ", " << v[i];cerr << "]" << endl;}
template<class T> void printWithIndex(VC < T > v) {cerr << "[";if (v.size()) cerr << "0:" <<  v[0];FOR(i, 1, v.size()) cerr << ", " << i << ":" <<  v[i];cerr << "]" << endl;}
inline VS splt(string s, char c = ',') {
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

#ifdef LOCAL
static bool LOG_DEBUG = true;
#else
static bool LOG_DEBUG = false;
#endif
/*! the message buffer length */
const int kPrintBuffer = 1 << 12;
inline void Printf(const char *fmt, ...) {
    if (LOG_DEBUG) {
        std::string msg(kPrintBuffer, '\0');
        va_list args;
        va_start(args, fmt);
        vsnprintf(&msg[0], kPrintBuffer, fmt, args);
        va_end(args);
        fprintf(stderr, "%s", msg.c_str());
    }
}

inline void Assert(bool exp, const char *fmt, ...) {
    if (!exp) {
        std::string msg(kPrintBuffer, '\0');
        va_list args;
        va_start(args, fmt);
        vsnprintf(&msg[0], kPrintBuffer, fmt, args);
        va_end(args);
        fprintf(stderr, "AssertError:%s\n", msg.c_str());
        exit(-1);
    }
}

inline double getTime() {
    timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

//
// ----------------------------
//
template <class T>
void concatenate(VC<T> &first, const VC<T> &second) {
    size_t size = second.size();
    if (first.size() < size) {
        // resize
        first.resize(size);
    }
    // update
    for (int i = 0; i < size; i++) {
        first[i] += second[i];
    }
}

template<class bidiiter> bidiiter
random_unique(bidiiter begin, bidiiter end, size_t num_random) {
    size_t left = std::distance(begin, end);
    while (num_random--) {
        bidiiter r = begin;
        std::advance(r, rand()%left);
        std::swap(*begin, *r);
        ++begin;
        --left;
    }
    return begin;
}

class RandomSample {
    // class members
    // generate "m_number" of data with the value within the range [0, m_max].
    int m_max;
    int m_number;
    
public:
    RandomSample(int max, int number) : m_max(max), m_number(number) {}
    
    inline VI get_sample_index() {
        // fill vector with indices
        VI re_res(m_max);
        for (int i = 0; i < m_max; ++i)
            re_res[i] = i;
        
        // suffle
        random_unique(re_res.begin(), re_res.end(), m_number);
        
        // resize vector
        re_res.resize(m_number);
        VI(re_res).swap(re_res);
        
        return re_res;
    }
};

struct GBTConfig {
    double sampling_size_ratio = 0.5;
    double learning_rate = 0.01;
    int tree_number = 130;
    int tree_min_nodes = 10;
    int tree_depth = 3;
};

struct Node {
    double m_node_value;
    int m_feature_index;
    double m_terminal_left;
    double m_terminal_right;
    
    // Each non-leaf node has a left child and a right child.
    Node *m_left_child = NULL;
    Node *m_right_child = NULL;
    
    // Construction function
    Node(double value, int feature_index, double value_left, double value_right) :
    m_node_value(value), m_feature_index(feature_index), m_terminal_left(value_left), m_terminal_right(value_right) {}
    
private:
    
    Node(Node const &); // non construction-copyable
    Node& operator=(Node const &); // non copyable
};

struct BestSplit {
    // the index of split feature
    int m_feature_index;
    // the calculated node value
    double m_node_value;
    // if false - split failed
    bool m_status;
    
    // construction function
    BestSplit() : m_feature_index(0.0), m_node_value(0.0), m_status(false) {}
};

struct SplitRes {
    VC<VD> m_feature_left;
    VC<VD> m_feature_right;
    double m_left_value = 0.0;
    double m_right_value = 0.0;
    VD m_obs_left;
    VD m_obs_right;
    
    // construction function
    SplitRes() : m_left_value(0.0), m_right_value(0.0) {}
};

struct ListData {
    double m_x;
    double m_y;
    
    ListData(double x, double y) : m_x(x), m_y(y) {}
    
    bool operator < (const ListData& str) const {
        return (m_x < str.m_x);
    }
};

typedef enum _TerminalType {
    AVERAGE, MAXIMAL
}TerminalType;

class RegressionTree {
private:
    // class members
    int m_min_nodes;
    int m_max_depth;
    int m_current_depth;
    TerminalType m_type;
    
public:
    // The root node
    Node *m_root = NULL;
    // The features importance per index
    VI features_importance;
    
    // construction function
    RegressionTree() : m_min_nodes(10), m_max_depth(3), m_current_depth(0), m_type(AVERAGE) {}
    
    // set parameters
    void setMinNodes(int min_nodes) {
        Assert(min_nodes > 3, "The number of terminal nodes is too small: %i", min_nodes);
        m_min_nodes = min_nodes;
    }
    
    void setDepth(int depth) {
        Assert(depth > 0, "Tree depth must be positive: %i", depth);
        m_max_depth = depth;
    }
    
    // get fit value
    double predict(const VD &feature_x) const{
        double re_res = 0.0;
        
        if (!m_root) {
            // failed in building the tree
            return re_res;
        }
        
        Node *current = m_root;
        
        while (true) {
            // current node information
            int c_feature_index = current->m_feature_index;
            double c_node_value = current->m_node_value;
            double c_node_left_value = current->m_terminal_left;
            double c_node_right_value = current->m_terminal_right;
            
            if (feature_x[c_feature_index] < c_node_value) {
                // we should consider left child
                current = current->m_left_child;
                
                if (!current) {
                    re_res = c_node_left_value;
                    break;
                }
            } else {
                // we should consider right child
                current = current->m_right_child;
                
                if (!current) {
                    re_res = c_node_right_value;
                    break;
                }
            }
        }
        
        return re_res;
    }
    
    /*
     *  The method to build regression tree
     */
    void buildRegressionTree(const VC<VD> &feature_x, const VD &obs_y) {
        size_t samples_num = feature_x.size();
        
        Assert(samples_num == obs_y.size() && samples_num != 0,
               "The number of samles does not match with the number of observations or the samples number is 0. Samples: %i", samples_num);
        
        Assert (m_min_nodes * 2 <= samples_num, "The number of samples is too small");
        
        size_t feature_dim = feature_x[0].size();
        features_importance.resize(feature_dim, 0);
        
        // build the regression tree
        buildTree(feature_x, obs_y);
    }
    
private:
    
    /*
     *  The following function gets the best split given the data
     */
    BestSplit findOptimalSplit(const VC<VD> &feature_x, const VD &obs_y) {
        
        BestSplit split_point;
        
        if (m_current_depth > m_max_depth) {
            return split_point;
        }
        
        size_t samples_num = feature_x.size();
        
        if (m_min_nodes * 2 > samples_num) {
            // the number of observations in terminals is too small
            return split_point;
        }
        size_t feature_dim = feature_x[0].size();
        
        
        double min_err = 0;
        int split_index = -1;
        double node_value = 0.0;
        
        // begin to get the best split information
        for (int loop_i = 0; loop_i < feature_dim; loop_i++){
            // get the optimal split for the loop_index feature
            
            // get data sorted by the loop_i-th feature
            VC<ListData> list_feature;
            for (int loop_j = 0; loop_j < samples_num; loop_j++) {
                list_feature.push_back(ListData(feature_x[loop_j][loop_i], obs_y[loop_j]));
            }
            
            // sort the list
            sort(list_feature.begin(), list_feature.end());
            
            // begin to split
            double sum_left = 0.0;
            double mean_left = 0.0;
            int count_left = 0;
            double sum_right = 0.0;
            double mean_right = 0.0;
            int count_right = 0;
            double current_node_value = 0;
            double current_err = 0.0;
            
            // initialize left
            for (int loop_j = 0; loop_j < m_min_nodes; loop_j++) {
                ListData fetched_data = list_feature[loop_j];
                sum_left += fetched_data.m_y;
                count_left++;
            }
            mean_left = sum_left / count_left;
            
            // initialize right
            for (int loop_j = m_min_nodes; loop_j < samples_num; loop_j++) {
                ListData fetched_data = list_feature[loop_j];
                sum_right += fetched_data.m_y;
                count_right++;
            }
            mean_right = sum_right / count_right;
            
            // calculate the current error
            // err = ||x_l - mean(x_l)||_2^2 + ||x_r - mean(x_r)||_2^2
            // = ||x||_2^2 - left_count * mean(x_l)^2 - right_count * mean(x_r)^2
            // = constant - left_count * mean(x_l)^2 - right_count * mean(x_r)^2
            // Thus, we only need to check "- left_count * mean(x_l)^2 - right_count * mean(x_r)^2"
            current_err = -1 * count_left * mean_left * mean_left - count_right * mean_right * mean_right;
            
            // current node value
            current_node_value = (list_feature[m_min_nodes].m_x + list_feature[m_min_nodes - 1].m_x) / 2;
            
            if (current_err < min_err && current_node_value != list_feature[m_min_nodes - 1].m_x) {
                split_index = loop_i;
                node_value = current_node_value;
                min_err = current_err;
            }
            
            // begin to find the best split point for the feature
            for (int loop_j = m_min_nodes; loop_j <= samples_num - m_min_nodes - 1; loop_j++) {
                ListData fetched_data = list_feature[loop_j];
                double y = fetched_data.m_y;
                sum_left += y;
                count_left++;
                mean_left = sum_left / count_left;
                
                
                sum_right -= y;
                count_right--;
                mean_right = sum_right / count_right;
                
                
                current_err = -1 * count_left * mean_left * mean_left - count_right * mean_right * mean_right;
                // current node value
                current_node_value = (list_feature[loop_j + 1].m_x + fetched_data.m_x) / 2;
                
                if (current_err < min_err && current_node_value != fetched_data.m_x) {
                    split_index = loop_i;
                    node_value = current_node_value;
                    min_err = current_err;
                }
                
            }
        }
        // set the optimal split point
        if (split_index == -1) {
            // failed to split data
            return split_point;
        }
        split_point.m_feature_index = split_index;
        split_point.m_node_value = node_value;
        split_point.m_status = true;
        
        return split_point;
    }
    
    /*
     *  Split data into the left node and the right node based on the best splitting
     *  point.
     */
    SplitRes splitData(const VC<VD> &feature_x, const VD &obs_y, const BestSplit &best_split) {
        
        SplitRes split_res;
        
        int feature_index = best_split.m_feature_index;
        double node_value = best_split.m_node_value;
        
        size_t samples_count = obs_y.size();
        for (int loop_i = 0; loop_i < samples_count; loop_i++) {
            VD ith_feature = feature_x[loop_i];
            if (ith_feature[feature_index] < node_value) {
                // append to the left feature
                split_res.m_feature_left.push_back(ith_feature);
                // observation
                split_res.m_obs_left.push_back(obs_y[loop_i]);
            } else {
                // append to the right
                split_res.m_feature_right.push_back(ith_feature);
                split_res.m_obs_right.push_back(obs_y[loop_i]);
            }
        }
        
        // update terminal values
        if (m_type == AVERAGE) {
            double mean_value = 0.0;
            for (double obsL : split_res.m_obs_left) {
                mean_value += obsL;
            }
            mean_value = mean_value / split_res.m_obs_left.size();
            split_res.m_left_value = mean_value;
            
            mean_value = 0.0;
            for (double obsR : split_res.m_obs_right) {
                mean_value += obsR;
            }
            mean_value = mean_value / split_res.m_obs_right.size();
            split_res.m_right_value = mean_value;
            
        } else if (m_type == MAXIMAL) {
            double max_value = 0.0;
            VD::iterator iter = split_res.m_obs_left.begin();
            if (++iter != split_res.m_obs_left.end()) {
                max_value = *iter;
            }
            
            while (++iter != split_res.m_obs_left.end()) {
                double sel_value = *iter;
                if (max_value < sel_value) {
                    max_value = sel_value;
                }
            }
            
            split_res.m_left_value = max_value;
            
            
            // right value
            max_value = 0.0;
            iter = split_res.m_obs_right.begin();
            if (++iter != split_res.m_obs_right.end()) {
                max_value = *iter;
            }
            
            while (++iter != split_res.m_obs_right.end()) {
                double sel_value = *iter;
                if (max_value < sel_value) {
                    max_value = sel_value;
                }
            }
            
            split_res.m_right_value = max_value;
            
        } else {
            // Unknown terminal type
            assert(false);
        }
        
        // return the result
        return split_res;
    }
    
    /*
     *  The following function builds a regression tree from data
     */
    Node* buildTree(const VC<VD> &feature_x, const VD &obs_y) {
        
        // obtain the optimal split point
        m_current_depth = m_current_depth + 1;
        
        BestSplit best_split = findOptimalSplit(feature_x, obs_y);
        
        if (!best_split.m_status) {
            if (m_current_depth > 0)
                m_current_depth = m_current_depth - 1;
            
            return NULL;
        }
        
        // update feature importance info
        features_importance[best_split.m_feature_index] += 1;
        
        // split the data
        SplitRes split_data = splitData(feature_x, obs_y, best_split);
        
        // append current value to tree
        Node *new_node = new Node(best_split.m_node_value, best_split.m_feature_index, split_data.m_left_value, split_data.m_right_value);
        
        if (!m_root) {
            m_root = new_node;
            m_current_depth = 0;
            // append left and right side
            m_root->m_left_child = buildTree(split_data.m_feature_left, split_data.m_obs_left); // left
            m_root->m_right_child = buildTree(split_data.m_feature_right, split_data.m_obs_right); // right
        } else {
            // append left and right side
            new_node->m_left_child = buildTree(split_data.m_feature_left, split_data.m_obs_left); // left
            new_node->m_right_child = buildTree(split_data.m_feature_right, split_data.m_obs_right); // right
        }
        if (m_current_depth > 0)
            m_current_depth--;
        
        return new_node;
    }
};

class PredictionForest {
public:
    // class members
    double m_init_value;
    // the tree forest
    VC<RegressionTree> m_trees;
    // the learning rate
    double m_combine_weight;
    
    // the OOB error value
    double oob_error;
    // the OOB samples size
    int oob_samples_size;
    
    // construction function
    PredictionForest(double learning_rate) : m_init_value(0.0), m_combine_weight(learning_rate) {}
    
    /**
     * The method to make prediction for estimate of function's value from provided features
     *
     * @param feature_x the features to use for prediction
     * @return the estimated function's value
     */
    double predict(const VD &feature_x) {
        double re_res = m_init_value;
        
        if (m_trees.size() == 0) {
            return re_res;
        }
        
        for (int i = 0; i < m_trees.size(); i++) {
            re_res += m_combine_weight * m_trees[i].predict(feature_x);
        }
        
        return re_res;
    }
    
    /**
     * Calculates importance of each feature in input samples
     */
    VI featureImportances() {
        VI importances;
        for (int i = 0; i < m_trees.size(); i++) {
            concatenate(importances, m_trees[i].features_importance);
        }
        return importances;
    }
};


class GradientBoostingMachine {
    // class members
    double m_sampling_size_ratio = 0.5;
    double m_learning_rate = 0.01;
    int m_tree_number = 100;
    
    // tree related parameters
    int m_tree_min_nodes = 10;
    int m_tree_depth = 3;
    
public:
    
    GradientBoostingMachine(double sample_size_ratio, double learning_rate,
                            int tree_number, int tree_min_nodes, int tree_depth) :
    m_sampling_size_ratio(sample_size_ratio), m_learning_rate(learning_rate), m_tree_number(tree_number),
    m_tree_min_nodes(tree_min_nodes), m_tree_depth(tree_depth) {
        // Check the validity of numbers
        Assert(sample_size_ratio > 0 && learning_rate > 0 && tree_number > 0 && tree_min_nodes >= 3 && tree_depth > 0,
               "Wrong parameters");
    }
    
    /**
     * Fits a regression function using the Gradient Boosting Tree method.
     * On success, return function; otherwise, return null.
     *
     * @param input_x the input features
     * @param input_y the ground truth values - one per features row
     */
    PredictionForest *train(const VC<VD> &input_x, const VD &input_y) {
        
        // initialize forest
        PredictionForest *res_fun = new PredictionForest(m_learning_rate);
        
        // get the samples number
        size_t samples_num = input_y.size();
        
        Assert(samples_num == input_x.size() && samples_num > 0,
               "Error: The input_x size should not be zero and should match the size of input_y");
        
        // holds indices of training data samples used for trees training
        // this will be used later for OOB error calculation
        VI used_indices(samples_num, -1);
        
        // get an initial guess of the function
        double mean_y = 0.0;
        for (double d : input_y) {
            mean_y += d;
        }
        mean_y = mean_y / samples_num;
        res_fun->m_init_value = mean_y;
        
        
        // prepare the iteration
        VD h_value(samples_num);
        // initialize h_value
        int index = 0;
        while (index < samples_num) {
            h_value[index] = mean_y;
            index += 1;
        }
        
        // begin the boosting process
        int iter_index = 0;
        while (iter_index < m_tree_number) {
            
            // calculate the gradient
            VD gradient;
            index = 0;
            for (double d : input_y) {
                gradient.push_back(d - h_value[index]);
                
                // next
                index++;
            }
            
            // begin to sample
            if (m_sampling_size_ratio < 0.99) {
                // sample without replacement
                
                // we need to sample
                RandomSample sampler((int)samples_num, (int) (m_sampling_size_ratio * samples_num));
                
                // get random index
                VI sampled_index = sampler.get_sample_index();
                
                // data for growing trees
                VC<VD> train_x;
                VD train_y;
                
                for (int sel_index : sampled_index) {
                    // assign value
                    train_y.push_back(gradient[sel_index]);
                    train_x.push_back(input_x[sel_index]);
                    
                    // mark index as used
                    used_indices[sel_index] = 1;
                }
                
                // fit a regression tree
                RegressionTree tree;
                
                if (m_tree_depth > 0) {
                    tree.setDepth(m_tree_depth);
                }
                
                if (m_tree_min_nodes > 0) {
                    tree.setMinNodes(m_tree_min_nodes);
                }
                
                tree.buildRegressionTree(train_x, train_y);
                
                // store tree information
                if (tree.m_root == NULL) {
                    // clear buffer
                    train_x.clear();
                    train_y.clear();
                    continue;
                }
                
                res_fun->m_trees.push_back(tree);
                
                // update h_value information, prepare for the next iteration
                int sel_index = 0;
                while (sel_index < samples_num) {
                    h_value[sel_index] += m_learning_rate * tree.predict(input_x[sel_index]);
                    sel_index++;
                }
                
            } else {
                // use all data
                // fit a regression tree
                RegressionTree tree;
                
                // set parameters if needed
                if (m_tree_depth > 0) {
                    tree.setDepth(m_tree_depth);
                }
                
                if (m_tree_min_nodes > 0) {
                    tree.setMinNodes(m_tree_min_nodes);
                }
                
                tree.buildRegressionTree(input_x, gradient);
                
                if (tree.m_root == NULL) {
                    // cannot update any more
                    break;
                }
                // store tree information
                res_fun->m_trees.push_back(tree);
                
                // update h_value information, prepare for the next iteration
                for (int loop_index = 0; loop_index < samples_num; loop_index++) {
                    h_value[loop_index] += m_learning_rate * tree.predict(input_x[loop_index]);
                }
            }
            
            // next iteration
            iter_index++;
        }
        
        // find OOB error
        VI oob_data;
        int i, sel_index;
        for (i = 0; i < samples_num; i++) {
            if (used_indices[i] < 0) {
                oob_data.push_back(i);
            }
        }
        double oob_error = 0.0, test_y;
        for (i = 0; i < oob_data.size(); i++) {
            sel_index = oob_data[i];
            test_y = res_fun->predict(input_x[sel_index]);
            oob_error += (input_y[sel_index] - test_y) * (input_y[sel_index] - test_y);
        }
        oob_error /= oob_data.size();
        
        // store OOB
        res_fun->oob_error = oob_error;
        res_fun->oob_samples_size = (int)oob_data.size();
        
        return res_fun;
    }
    
    PredictionForest *learnGradientBoostingRanker(const VC<VD> &input_x, const VC<VD> &input_y, const double tau) {
        PredictionForest *res_fun = new PredictionForest(m_learning_rate);
        
        size_t feature_num = input_x.size();
        
        Assert(feature_num == input_y.size() && feature_num > 0,
               "The size of input_x should be the same as the size of input_y");
        
        VD h_value_x(feature_num, 0);
        VD h_value_y(feature_num, 0);
        
        int iter_index = 0;
        while (iter_index < m_tree_number) {
            
            // in the boosting ranker, randomly select half samples without replacement in each iteration
            RandomSample sampler((int)feature_num, (int) (0.5 * feature_num));
            
            // get random index
            VI sampled_index = sampler.get_sample_index();
            
            VC<VD> gradient_x;
            VD gradient_y;
            
            for (int i = 0; i < sampled_index.size(); i++) {
                int sel_index = sampled_index[i];
                
                gradient_x.push_back(input_x[sel_index]);
                gradient_x.push_back(input_y[sel_index]);
                
                // get sample data
                if (h_value_x[sel_index] < h_value_y[sel_index] + tau) {
                    double neg_gradient = h_value_y[sel_index] + tau - h_value_x[sel_index];
                    gradient_y.push_back(neg_gradient);
                    gradient_y.push_back(-1 * neg_gradient);
                } else {
                    gradient_y.push_back(0.0);
                    gradient_y.push_back(0.0);
                }
                //                cerr << "sel_index: " << sel_index << endl;
            }
            
            // fit a regression tree
            RegressionTree tree;
            //            tree.m_type = MAXIMAL;
            
            tree.buildRegressionTree(gradient_x, gradient_y);
            
            // store tree information
            if (tree.m_root == NULL) {
                continue;
            }
            
            // update information
            res_fun->m_trees.push_back(tree);
            
            double err = 0.0;
            
            for (int loop_index = 0; loop_index < feature_num; loop_index++) {
                h_value_x[loop_index] += m_learning_rate * tree.predict(input_x[loop_index]);
                h_value_y[loop_index] += m_learning_rate * tree.predict(input_y[loop_index]);
                
                if (h_value_x[loop_index] < h_value_y[loop_index] + tau) {
                    err += (h_value_x[loop_index] - h_value_y[loop_index] - tau) *
                    (h_value_x[loop_index] - h_value_y[loop_index] - tau);
                }
            }
            
            iter_index += 1;
        }
        
        
        
        return res_fun;
    }
};

//
// ----------------------------
//
class Matrix {
    
protected:
    size_t m, n;
    
    
public:
    VVD A;
    
    Matrix(const size_t rows, const size_t cols) {
        m = rows;
        n = cols;
        for (int i = 0; i < m; i++) {
            VD row(n, 0);
            A.push_back(row);
        }
    }
    
    Matrix(const VVD &arr) {
        m = arr.size();
        n = arr[0].size();
        for (int i = 0; i < m; i++) {
            assert(arr[i].size() == n);
        }
        A = arr;
    }
    
    size_t rows() const {
        return m;
    }
    
    size_t cols() const {
        return n;
    }
    
    Matrix& subMatrix(int i0, int i1, int j0, int j1) const {
        assert(i0 >= 0 && i0 <= i1 && i1 < m && j0 >= 0 && j0 <= j1 && j1 < n);
        Matrix *X = new Matrix(i1 - i0 + 1, j1 - j0 + 1);
        for (int i = i0; i <= i1; i++) {
            for (int j = j0; j <= j1; j++) {
                X->A[i - i0][j - j0] = A[i][j];
            }
        }
        return *X;
    }
    
    void columnToArray(const int col, VD &vals) const {
        assert(col < n);
        vals.resize(m, 0);
        for (int i = 0; i < m; i++) {
            vals[i] = A[i][col];
        }
    }
    
    void addRow(const VD &row) {
        assert(row.size() == n);
        A.push_back(row);
        // adjust row counts
        m = A.size();
    }
    
    void addColumn(const VD &col) {
        assert(col.size() == m);
        for (int i = 0; i < m; i++) {
            A[i].push_back(col[i]);
        }
        // inclrease column counts
        n += 1;
    }
    
    void concat(const Matrix &mat) {
        assert(mat.rows() == m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < mat.cols(); j++) {
                A[i].push_back(mat[i][j]);
            }
        }
        // inclrease column counts
        n += mat.cols();
    }
    
    Matrix& transpose() {
        Matrix *X = new Matrix(n, m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X->A[j][i] = A[i][j];
            }
        }
        return *X;
    }
    
    Matrix& operator=(const Matrix &B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->A[i][j] = B.A[i][j];
            }
        }
        return *this;
    }
    
    double& operator()(const int i, const int j) {
        assert( i >= 0 && i < m && j >=0 && j < n);
        return A[i][j];
    }
    double operator()(const int i, const int j) const{
        assert( i >= 0 && i < m && j >=0 && j < n);
        return A[i][j];
    }
    VD& operator[](const int row) {
        assert( row >= 0 && row < m);
        return A[row];
    }
    VD operator[](const int row) const{
        assert( row >= 0 && row < m);
        return A[row];
    }
    
    Matrix operator+(const Matrix& B) const {
        checkMatrixDimensions(B);
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X(i, j) = this->A[i][j] + B.A[i][j];
            }
        }
        return X;
    }
    
    Matrix& operator+=(const Matrix &B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->A[i][j] = this->A[i][j] + B.A[i][j];
            }
        }
        return *this;
    }
    
    Matrix operator-(const Matrix &B) const {
        checkMatrixDimensions(B);
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X(i, j) = this->A[i][j] - B.A[i][j];
            }
        }
        return X;
    }
    
    Matrix& operator-=(const Matrix &B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->A[i][j] = this->A[i][j] - B.A[i][j];
            }
        }
        return *this;
    }
    
    Matrix operator*(const Matrix &B) const {
        checkMatrixDimensions(B);
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X(i, j) = this->A[i][j] * B.A[i][j];
            }
        }
        return X;
    }
    
    Matrix& operator*=(const Matrix &B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->A[i][j] = this->A[i][j] * B.A[i][j];
            }
        }
        return *this;
    }
    
    Matrix operator*(const double s) const {
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X(i, j) = s * this->A[i][j];
            }
        }
        return X;
    }
    
    Matrix& operator*=(const double s) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->A[i][j] = s * this->A[i][j];
            }
        }
        return *this;
    }
    
    Matrix operator/(const Matrix &B) const {
        checkMatrixDimensions(B);
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X(i, j) = this->A[i][j] / B.A[i][j];
            }
        }
        return X;
    }
    
    Matrix& operator/=(const Matrix &B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->A[i][j] = this->A[i][j] / B.A[i][j];
            }
        }
        return *this;
    }
    
    Matrix& operator/=(const double s) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->A[i][j] =  this->A[i][j] / s;
            }
        }
        return *this;
    }
    
    Matrix operator/(const double s) const {
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X(i, j) = this->A[i][j] / s;
            }
        }
        return X;
    }
    
    bool operator==(const Matrix &B) const {
        if (m != B.m && n != B.n) {
            return false;
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (A[i][j] != B.A[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }
    
    bool similar(const Matrix &B, double diff) {
        if (m != B.m && n != B.n) {
            return false;
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (abs(A[i][j] - B.A[i][j]) > diff) {
                    return false;
                }
            }
        }
        return true;
    }
    
    Matrix& matmul(const Matrix &B) const {
        // Matrix inner dimensions must agree.
        assert (B.m == n);
        
        Matrix *X = new Matrix(m, B.n);
        double Bcolj[n];
        for (int j = 0; j < B.n; j++) {
            for (int k = 0; k < n; k++) {
                Bcolj[k] = B.A[k][j];
            }
            for (int i = 0; i < m; i++) {
                VD Arowi = A[i];
                double s = 0;
                for (int k = 0; k < n; k++) {
                    s += Arowi[k] * Bcolj[k];
                }
                X->A[i][j] = s;
            }
        }
        return *X;
    }
    
    
    
protected:
    void checkMatrixDimensions(Matrix B) const {
        assert (B.m != m || B.n != n);
    }
};

//
// -----------------------------------------
//

struct RRConfig {
    bool useBootstrap;
    int bootstrapSamples;
    int bootstrapMode;
    int bootstrapIterations;
    int regressionIterations;
    int regressionMode;
    
    int approxSamples;
    int approxMode;
    int approxIterations;
};

#define VERBOSE 0
class RidgeRegression {
    // regression coefficients
    VD regrCoef;
    // initial regression coefficients (usually 0)
    VD regrCoefSeed;
    // residual errors per observation
    VD obsErrors;
    
    // the OOB errors calculated if bootstrap used
    VD oobErrors;
    
    // the configuration
    RRConfig config;
    
    // the number of observations/samples
    size_t obs = -1;
    // the number of features per observation
    size_t var = -1;
    
    // flag to indicate that data structures initialize
    int init = -1;
    // flag to indicate that approximate regression coeff was calculated
    int seed = -1;
    
    // the calculated MSE after regression complete
    double MSE = 0;
    // the initial MSE before regresion starts
    double MSE_init = 0;
    
public:
    RidgeRegression(RRConfig conf) : config(conf) {}

    void train(const VVD &train, const VD &check) {
        Assert(train.size() == check.size(), "Samples size should be equal to observations size! Samples: %lu, observations: %lu", train.size(), check.size());
        // find number of variables and observations
        var = train[0].size();
        obs = train.size();
        
        initDataStructures();
        if (config.useBootstrap) {
            bootstrap(train, check, config.bootstrapSamples, config.bootstrapMode, config.bootstrapIterations);
            regress(train, check, config.regressionMode, config.regressionIterations, 1);
        } else {
            initFeaturesSeed(train, check, config.approxMode, config.approxSamples, config.approxIterations);
            regress(train, check, config.regressionMode, config.regressionIterations, 1);
        }
    }

    double predict(const VD &features) {
        Assert(features.size() == var, "Features size should be equal to train features size, but was: %li", features.size());
        double res = 0;
        for (int i = 0; i < features.size(); i++) {
            res += features[i] * regrCoef[i];
        }
        return res;
    }
    
private:

    void initFeaturesSeed(const VVD &train, const VD &check, const long mode, const long nvalid, const long niter) {
        // initialize data structure
        if (seed==1) {
            regrCoefSeed.clear();
        } else {
            seed=1;
        }
        regrCoefSeed.resize(var, 0);
        
        double nsvar, nsvar_max = -1;
        for (int n = 0; n < nvalid; n++) {
            nsvar = regress(train, check, mode, niter, 0);
            if (nsvar > nsvar_max) {
                nsvar_max = nsvar;
                for (int k = 0; k < var; k++) {
                    regrCoefSeed[k] = regrCoef[k];
                }
            }
        }
    }

    void bootstrap(const VVD &train, const VD &check, const long nsample, const long mode, const long niter) {
        /* need to run Regress_init first if the data set is new */
        Assert(init == 1, "Must run initDataStructures() fisrt.\n");
        
        // initialize data structure
        if (seed==1) {
            regrCoefSeed.clear();
        } else {
            seed=1;
        }
        regrCoefSeed.resize(var, 0);
        
        VVD bootTrain, bootTest;
        VD bootCheck, bootCheckTest;
        
        // the best found regression coefficients
        VD bestCoef(var, 0);
        
        // array to store selected indices
        VI pick(obs, 0);
        int k, idx, m, oi;
        double oobMSE, oobME, minOOBMSE = numeric_limits<double>::max();
        for (int n = 0; n < nsample; n++) {
            // pick up random observations
            for (k = 0; k < obs; k++) { pick[k] = 0; }
            for (k = 0; k < obs; k++) {
                idx = rand() % obs;
                pick[idx]++;
            }
            
            // create subsample
            for (m = 0; m < obs; m++) {
                if (pick[m] > 0) {
                    // save pick[m] copies of row in data
                    for (k = 0; k < pick[m]; k++) {
                        bootTrain.push_back(train[m]);
                        bootCheck.push_back(check[m]);
                    }
                } else {
                    // save current row as test sample
                    bootTest.push_back(train[m]);
                    bootCheckTest.push_back(check[m]);
                }
            }
            // do regression
            initDataStructures();
            regress(bootTrain, bootCheck, mode, niter, 0);
            
            // find OOB error
            oobMSE = 0;
            for (oi = 0; oi < bootTest.size(); oi++) {
                oobME = predict(bootTest[oi]) - bootCheckTest[oi];
                oobMSE += oobME * oobME;
            }
            oobMSE = sqrt(oobMSE /(double)bootTest.size());
            oobErrors.push_back(oobMSE);
            if (oobMSE < minOOBMSE) {
                oobMSE = minOOBMSE;
                // store current regression coefficients
                bestCoef.swap(regrCoef);
            }
        }
        
        // store best regression coefficients as train results
        regrCoefSeed.swap(bestCoef);
    }

    double regress(const VVD &features, const VD &dv, const long mode, const long niter, const long seed_flag) {
        /* need to run Regress_init first if the data set is new */
        Assert(init == 1, "Must run initDataStructures() fisrt.\n");
        
        int k, l, i, iter;
        double xd, sp, lambda, val, e_new, resvar = 0;
        
        // calculate initial MSE
        MSE_init = 0;
        for (k = 0; k < obs; k++) {
            val = dv[k];
            MSE_init += val * val;
            obsErrors[k] = val;
        }
        MSE_init = sqrt(MSE_init / obs);
        
        // clear regression coefficients
        for (k = 1; k < var; k++) {
            regrCoef[k] = 0;
        }
        
        // if seed=1 uses initial regressors
        if (seed_flag == 1) {
            Assert(seed == 1, "Must run initFeaturesSeed() first.\n");
            
            for (k = 0; k < var; k++) {
                regrCoef[k] = regrCoefSeed[k];
            }
            
            for (i = 0; i < obs; i++) {
                e_new = 0;
                for (k = 0; k < var; k++) {
                    val = features[i][k];
                    e_new += regrCoef[k] * val;
                }
                // find error
                obsErrors[i] = dv[i] - e_new;
            }
        }
        
        /*
         regression
         */
        for (iter = 0; iter < niter; iter++) {
            if (mode == 0 || mode == 3) {
                // 0: visits each variable sequentially starting with first
                l = iter % var;
            } else {
                // 1: visits variables in random order; should be the default mode
                // 2: same as mode = 1, but lambda not randomized
                l = rand() % var;
            }
            
            xd = 0; sp = 0;
            for (k = 0; k < obs; k++) {
                xd += features[k][l] * features[k][l];
                sp += features[k][l] * obsErrors[k];
            }
            Assert(xd != 0, "Empty column found at index: %i", l);
            
            lambda = sp / xd;
            if (mode == 1) {
                lambda = lambda * rand() / (double)RAND_MAX;
            }
            
            // update error
            MSE = 0;
            for (k = 0; k < obs; k++) {
                e_new = obsErrors[k] - lambda * features[k][l];
                MSE += e_new * e_new;
                obsErrors[k] = e_new;
            }
            regrCoef[l] += lambda;
            
            /*
             save results, compute resvar
             */
            
            if (iter % 10 == VERBOSE || iter == niter-1) {
                MSE = sqrt(MSE / obs);
                resvar = 1 - MSE / MSE_init;
                if (LOG_DEBUG) {
                    Printf("REGRESS %d\t%d\t%f\t%f\t", iter, l, lambda, resvar);
                    print(regrCoef);
                }
            }
        }
        return resvar;
    }
    
    /**
     * Reinitialize internal data structures.
     */
    void initDataStructures() {
        // clear data if needed
        if (init == 1) {
            regrCoef.clear();
            obsErrors.clear();
        } else  {
            init = 1;
        }
        
        // resize internal data structures
        regrCoef.resize(var, 0);
        obsErrors.resize(obs, 0);
    }
    
};

//
// ---------------------------------------------------------
//
struct Entry {
    constexpr static const double NVL = -1000.0;
    
    int subjid;
    double agedays;
    double wtkg;
    double htcm;
    int lencm;
    double bmi;
    double waz;
    double haz;
    double whz;
    double baz;
    int siteid;
    int sexn;// 1 = Male, 2 = Female
    int feedingn;// Breast feeding category 1 = Exclusively breast fed, 2 = Exclusively formula fed, 3 = Mixture breast/formula fed, 90 = Unknown
    int gagebrth;
    int birthwt;
    int birthlen;
    int apgar1;
    int apgar5;
    int mage;
    int demo1n;
    int mmaritn;// 1 = Married, 2 = Common law, 3 = Separated, 4 = Divorced, 5 = Widowed, 6 = Single
    int mcignum;
    int parity;
    int gravida;
    int meducyrs;
    int demo2n;
    int geniq;
    
    static double stof(string s) {
        return s == "NA" ? NVL : atof(s.c_str());
    }
    
    static double convertx(double s) {
        return s == NVL ? 0.0 : s;
    }
    
    Entry(){}
    Entry(string s, int scenario) {
        VS v = splt(s, ',');
        int pos = 0;
        bool hasIQ = false;
        switch(scenario) {
            case 0:
                Assert(v.size() == 7 || v.size() == 8, "Features size for first scenario, expected 7/8, but was: %lu", v.size());
                pos = parseScenario0(v);
                hasIQ = (v.size() == 8);
                break;
                
            case 1:
                Assert(v.size() == 16 || v.size() == 17, "Features size for second scenario, expected 16/17, but was: %lu", v.size());
                pos = parseScenario0(v);
                pos = parseScenario1(v, pos);
                hasIQ = (v.size() == 17);
                break;
                
            case 2:
                Assert(v.size() == 26 || v.size() == 27, "Features size for third scenario, expected 26/27, but was: %lu", v.size());
                pos = parseScenario0(v);
                pos = parseScenario1(v, pos);
                pos = parseScenario2(v, pos);
                hasIQ = (v.size() == 27);
                break;
        }
        // store IQ
        if (hasIQ) {
            geniq = (int)stof(v[pos]);
        }
    }
    
    int parseScenario0(const VS &v) {
        int pos = 0;
        subjid = (int)stof(v[pos++]);
        sexn = (int)stof(v[pos++]);
        gagebrth = (int)stof(v[pos++]);
        birthwt = (int)stof(v[pos++]);
        birthlen = (int)stof(v[pos++]);
        apgar1 = (int)stof(v[pos++]);
        apgar5 = (int)stof(v[pos++]);
        
        return pos;
    }
    
    int parseScenario1(const VS &v, int pos) {
        agedays = (int)stof(v[pos++]);
        wtkg = stof(v[pos++]);
        htcm = stof(v[pos++]);
        lencm = stof(v[pos++]);
        bmi = stof(v[pos++]);
        waz = stof(v[pos++]);
        haz = stof(v[pos++]);
        whz = stof(v[pos++]);
        baz = stof(v[pos++]);
        
        return pos;
    }
    
    int parseScenario2(const VS &v, int pos) {
        siteid = (int)stof(v[pos++]);
        feedingn = (int)stof(v[pos++]);
        mage = (int)stof(v[pos++]);
        demo1n = (int)stof(v[pos++]);
        mmaritn = (int)stof(v[pos++]);
        mcignum = (int)stof(v[pos++]);
        parity = (int)stof(v[pos++]);
        gravida = (int)stof(v[pos++]);
        meducyrs = (int)stof(v[pos++]);
        demo2n = (int)stof(v[pos++]);
        
        return pos;
    }
};


VVE readEntries(const VS &data, const int scenario) {
    VVE rv;
    int lastIndex = -1, index = -1;
    for (int i = 0; i < data.size(); i++) {
        string line = data[i];
        VS s = splt(line);
        int subjid = atof(s[0].c_str());
        
        if (subjid - 1 != lastIndex) {
            VE samples;
            rv.push_back(samples);
            lastIndex = subjid - 1;
            index ++;
        }
        
        // add entry
        rv[index].push_back(Entry(line, scenario));
    }
    return rv;
}

Matrix& prepareScenario1Features(const VVE &data) {
    // sexn, gagebrth, birthwt, birthlen, apgar1, apgar5
    int gagebrth = 0, birthwt = 0, birthlen = 0, apgar1 = 0, apgar5 = 0;
    VI counts(5, 0);
    for (const VE &smpls : data) {
        if (smpls[0].gagebrth != Entry::NVL) {
            gagebrth += smpls[0].gagebrth;
            counts[0]++;
        }
        if (smpls[0].birthwt != Entry::NVL) {
            birthwt += smpls[0].birthwt;
            counts[1]++;
        }
        if (smpls[0].birthlen != Entry::NVL) {
            birthlen += smpls[0].birthlen;
            counts[2]++;
        }
        if (smpls[0].apgar1 != Entry::NVL) {
            apgar1 += smpls[0].apgar1;
            counts[3]++;
        }
        if (smpls[0].apgar5 != Entry::NVL) {
            apgar5 += smpls[0].apgar5;
            counts[4]++;
        }
    }
    // find mean average
    gagebrth    /= counts[0];
    birthwt     /= counts[1];
    birthlen    /= counts[2];
    apgar1      /= counts[3];
    apgar5      /= counts[4];
    
    // build data matrix
    VVD features;
    for (const VE &smpls : data) {
        VD row;
        row.push_back(smpls[0].sexn);
        row.push_back(smpls[0].gagebrth == Entry::NVL ? gagebrth : smpls[0].gagebrth);
        row.push_back(smpls[0].birthwt == Entry::NVL ? birthwt : smpls[0].birthwt);
        row.push_back(smpls[0].birthlen == Entry::NVL ? birthlen : smpls[0].birthlen);
        row.push_back(smpls[0].apgar1 == Entry::NVL ? apgar1 : smpls[0].apgar1);
        row.push_back(smpls[0].apgar5 == Entry::NVL ? apgar5 : smpls[0].apgar5);
        features.push_back(row);
    }
    
    Matrix *m = new Matrix(features);
    return *m;
}

double predictWTKG(const VE &smpls, const int pos) {
    int low = pos, high = pos;
    while (low > 0 && smpls[low].wtkg == Entry::NVL) low--;
    while (high < smpls.size() && smpls[high].wtkg == Entry::NVL) high++;
    if (low < 0 || high >=  smpls.size()) {
        return Entry::NVL;
    }

    double w = (smpls[pos].agedays - smpls[low].agedays) / (smpls[high].agedays - smpls[low].agedays);
    if (smpls[low].wtkg == Entry::NVL || smpls[high].wtkg == Entry::NVL) {
        return Entry::NVL;
    }
    double wtkg = smpls[low].wtkg * (1 - w) + smpls[high].wtkg * w;
    return wtkg;
}

double predictL(const VE &smpls, const int pos) {
    int low = pos, high = pos;
    while (low > 0 && smpls[low].htcm == Entry::NVL && smpls[low].lencm == Entry::NVL) low--;
    while (high < smpls.size() && smpls[high].htcm == Entry::NVL && smpls[high].lencm == Entry::NVL) high++;
    if (low < 0 || high >=  smpls.size()) {
        return Entry::NVL;
    }
    
    double w = (smpls[pos].agedays - smpls[low].agedays) / (smpls[high].agedays - smpls[low].agedays);
    double lowL = (smpls[low].htcm == Entry::NVL ? smpls[low].lencm : smpls[low].htcm);
    double highL = (smpls[high].htcm == Entry::NVL ? smpls[high].lencm : smpls[high].htcm);
    
    if (lowL == Entry::NVL || highL == Entry::NVL) {
        return Entry::NVL;
    }

    double L = lowL * (1 - w) + highL * w;
    return L;
}

inline double safeDiv(const double a, const double b, const double d) {
    if (b == 0) {
        return d;
    } else {
        return a / b;
    }
}



Matrix& prepareScenario2Features(const VVE &data) {
    // agedays, wtkg, htcm, lencm, bmi, waz, haz, whz, baz
    VVD features;
    for (const VE &smpls : data) {
        
        double wtkg = 0, len = 0, bmi = 0, waz = 0, haz = 0, whz = 0, baz = 0;
        VI counts(7, 0);
        for (int i = 0; i < smpls.size(); i++) {
            double w, l;
            // add wtkg
            if (smpls[i].wtkg != Entry::NVL) {
                w = smpls[i].wtkg;
            } else {
                w = predictWTKG(smpls, i);
            }
            if (w != Entry::NVL) {
                wtkg += w;
                counts[0]++;
            }

            // add lenght
            if (smpls[i].htcm != Entry::NVL) {
                l = smpls[i].htcm;
            } else if (smpls[i].lencm != Entry::NVL) {
                l = smpls[i].lencm;
            } else {
                l = predictL(smpls, i);
            }
            if (l != Entry::NVL) {
                len += l;
                counts[1]++;
            }
            
            // add bmi
            if (smpls[i].bmi != Entry::NVL) {
                bmi += smpls[i].bmi;
            } else if (w != Entry::NVL && l != Entry::NVL){
                bmi += w / (l * l / 10000);
            }
            counts[2]++;
            
            if (smpls[i].waz != Entry::NVL) {
                waz += smpls[i].waz;
                counts[3]++;
            }
            
            if (smpls[i].haz != Entry::NVL) {
                haz += smpls[i].haz;
                counts[4]++;
            }
            
            if (smpls[i].whz != Entry::NVL) {
                whz += smpls[i].whz;
                counts[5]++;
            }
            
            if (smpls[i].baz != Entry::NVL) {
                baz += smpls[i].baz;
                counts[6]++;
            }
        }
        VD row = {safeDiv(wtkg, counts[0], 0), safeDiv(len, counts[1], 0), safeDiv(bmi, counts[2], 0), safeDiv(waz, counts[3], 0), safeDiv(haz, counts[4], 0), safeDiv(whz, counts[5], 0), safeDiv(baz, counts[6], 0)};
#ifdef LOCAL
        for (int i = 0; i < row.size(); i++) {
            if (isnan(row[i]) == true) {
                Printf("NaN at: %i in sample: %i\n", i, smpls[0].subjid);
                Assert(isnan(row[i]) == true, "Found NaN for index: %i in sample: %i\n", i, smpls[0].subjid);
            }
            
        }
#endif
        features.push_back(row);
    }
    Matrix *m = new Matrix(features);
    return *m;
}

Matrix& prepareScenario3Features(const VVE &data) {
    // siteid, feedingn, mage, demo1n, mmaritn, mcignum, parity, gravida, meducyrs, demo2n
    double feedingn = 0, mage = 0, demo1n = 0, mmaritn = 0, mcignum = 0, parity = 0, gravida = 0, meducyrs = 0, demo2n = 0;
    VI counts(9, 0);
    for (const VE &smpls : data) {
        int idx = 0;
        if (smpls[0].feedingn != Entry::NVL && smpls[0].feedingn != 90) {
            feedingn += smpls[0].feedingn;
            counts[idx++]++;
        }
        if (smpls[0].mage != Entry::NVL) {
            mage = smpls[0].mage;
            counts[idx++]++;
        }
        if (smpls[0].demo1n != Entry::NVL) {
            demo1n = smpls[0].demo1n;
            counts[idx++]++;
        }
        if (smpls[0].mmaritn != Entry::NVL) {
            mmaritn = smpls[0].mmaritn;
            counts[idx++]++;
        }
        if (smpls[0].mcignum != Entry::NVL) {
            mcignum = smpls[0].mcignum;
            counts[idx++]++;
        }
        if (smpls[0].parity != Entry::NVL) {
            parity = smpls[0].parity;
            counts[idx++]++;
        }
        if (smpls[0].gravida != Entry::NVL) {
            gravida = smpls[0].gravida;
            counts[idx++]++;
        }
        if (smpls[0].meducyrs != Entry::NVL) {
            meducyrs = smpls[0].meducyrs;
            counts[idx++]++;
        }
        if (smpls[0].demo2n != Entry::NVL) {
            demo2n = smpls[0].demo2n;
            counts[idx++]++;
        }
    }
    feedingn = safeDiv(feedingn,    counts[0], 0);
    mage = safeDiv(mage,            counts[1], 0);
    demo1n = safeDiv(demo1n,        counts[2], 0);
    mmaritn = safeDiv(mmaritn,      counts[3], 0);
    mcignum = safeDiv(mcignum,      counts[4], 0);
    parity = safeDiv(parity,        counts[5], 0);
    gravida = safeDiv(gravida,      counts[6], 0);
    meducyrs = safeDiv(meducyrs,    counts[7], 0);
    demo2n = safeDiv(demo2n,        counts[8], 0);
    
    // build data matrix
    VVD features;
    for (const VE &smpls : data) {
        VD row;
        row.push_back(smpls[0].siteid);
        row.push_back(smpls[0].feedingn != Entry::NVL ? smpls[0].feedingn : feedingn);
        row.push_back(smpls[0].mage != Entry::NVL ? smpls[0].mage : mage);
        row.push_back(smpls[0].demo1n != Entry::NVL ? smpls[0].demo1n : demo1n);
        row.push_back(smpls[0].mmaritn != Entry::NVL ? smpls[0].mmaritn : mmaritn);
        row.push_back(smpls[0].mcignum != Entry::NVL ? smpls[0].mcignum : mcignum);
        row.push_back(smpls[0].parity != Entry::NVL ? smpls[0].parity : parity);
        row.push_back(smpls[0].gravida != Entry::NVL ? smpls[0].gravida : gravida);
        row.push_back(smpls[0].meducyrs != Entry::NVL ? smpls[0].meducyrs : meducyrs);
        row.push_back(smpls[0].demo2n != Entry::NVL ? smpls[0].demo2n : demo2n);
        
        features.push_back(row);
    }
    Matrix *m = new Matrix(features);
    return *m;
}

class ChildStuntedness5 {
    size_t X;
    size_t Y;
    
public:
    /**
     * @param testType The testType parameter will be 0, 1, or 2, to indicate Example, Provisional, or System test
     * @param scenario The scenario parameter is also 0, 1, or 2, referring to the three scenarios listed above.
     */
    VD predict(const int testType, const int scenario, const VS &training, const VS &testing) {
        X = training.size();
        Y = testing.size();
        
        fprintf(stderr, "Test type: %i, scenario: %i, training size: %lu, test size: %lu\n", testType, scenario, X, Y);
        
        VVE trainEntries = readEntries(training, scenario);
        VVE testEntries = readEntries(testing, scenario);
        
        VD res;
        if (scenario == 0) {
            Matrix trainFeatures = prepareScenario1Features(trainEntries);
            Matrix testFeatures = prepareScenario1Features(testEntries);
            
            res = rankScenario1(trainFeatures, trainEntries, testFeatures);
        } else if (scenario == 1) {
            Matrix trainFeatures0 = prepareScenario1Features(trainEntries);
            Matrix testFeatures0 = prepareScenario1Features(testEntries);
            
            Matrix trainFeatures1 = prepareScenario2Features(trainEntries);
            Matrix testFeatures1 = prepareScenario2Features(testEntries);
            
            res = rankScenario2(trainFeatures0, trainFeatures1, trainEntries, testFeatures0, testFeatures1);
        } else {
            Matrix trainFeatures0 = prepareScenario1Features(trainEntries);
            Matrix testFeatures0 = prepareScenario1Features(testEntries);
            
            Matrix trainFeatures1 = prepareScenario2Features(trainEntries);
            Matrix testFeatures1 = prepareScenario2Features(testEntries);
            
            Matrix trainFeatures2 = prepareScenario3Features(trainEntries);
            Matrix testFeatures2 = prepareScenario3Features(testEntries);
            
            res = rankScenario3(trainFeatures0, trainFeatures1, trainFeatures2, trainEntries, testFeatures0, testFeatures1, testFeatures2);
        }

        return res;
    }
    
private:
    VD rankScenario3(const Matrix &trainFeatures0, const Matrix &trainFeatures1, const Matrix &trainFeatures2, const VVE &trainEntries,
                     const Matrix &testFeatures0, const Matrix &testFeatures1, const Matrix &testFeatures2) {
        cerr << "=========== Rank for scenario 3 ===========" << endl;
        
        double startTime = getTime();
        
        
        VD dv;
        for (const VE &smpls : trainEntries) {
            for (int i = 0; i < smpls.size(); i++) {
                if (smpls[i].geniq != Entry::NVL) {
                    dv.push_back(smpls[i].geniq);
                }
            }
        }
        
        // merge data
        Matrix trainFeatures = trainFeatures0;
        trainFeatures.concat(trainFeatures1);
        trainFeatures.concat(trainFeatures2);
        
        Matrix testFeatures = testFeatures0;
        testFeatures.concat(testFeatures1);
        testFeatures.concat(testFeatures2);
        
        VD res;
        GBTConfig conf;
        conf.sampling_size_ratio = 0.5;
        conf.learning_rate = 0.001;
        conf.tree_min_nodes = 10;
        conf.tree_depth = 7;
        conf.tree_number = (int)trainFeatures1.rows();
        
        GradientBoostingMachine tree(conf.sampling_size_ratio, conf.learning_rate, conf.tree_number, conf.tree_min_nodes, conf.tree_depth);
        PredictionForest *predictor = tree.train(trainFeatures.A, dv);
        
        // predict
        for (int i = 0; i < testFeatures.rows(); i++) {
            res.push_back(predictor->predict(testFeatures[i]));
        }
        
        print(res);
        
        double finishTime = getTime();
        
        Printf("Rank time: %f\n", finishTime - startTime);
        
        return res;
    }
    
    VD rankScenario2(const Matrix &trainFeatures0, const Matrix &trainFeatures1, const VVE &trainEntries,  const Matrix &testFeatures0, const Matrix &testFeatures1) {
        cerr << "=========== Rank for scenario 2 ===========" << endl;
        
        double startTime = getTime();
        
        
        VD dv;
        for (const VE &smpls : trainEntries) {
            for (int i = 0; i < smpls.size(); i++) {
                if (smpls[i].geniq != Entry::NVL) {
                    dv.push_back(smpls[i].geniq);
                }
            }
        }
        
        // merge data
        Matrix trainFeatures = trainFeatures0;
        trainFeatures.concat(trainFeatures1);
        
        Matrix testFeatures = testFeatures0;
        testFeatures.concat(testFeatures1);
        

        VD res;
        GBTConfig conf;
        conf.sampling_size_ratio = 0.5;
        conf.learning_rate = 0.001;
        conf.tree_min_nodes = 10;
        conf.tree_depth = 7;
        conf.tree_number = (int)trainFeatures1.rows();
        
        GradientBoostingMachine tree(conf.sampling_size_ratio, conf.learning_rate, conf.tree_number, conf.tree_min_nodes, conf.tree_depth);
        PredictionForest *predictor = tree.train(trainFeatures.A, dv);
        
        // predict
        for (int i = 0; i < testFeatures.rows(); i++) {
            res.push_back(predictor->predict(testFeatures[i]));
        }
        
        print(res);
        
        double finishTime = getTime();
        
        Printf("Rank time: %f\n", finishTime - startTime);
        
        return res;
    }
    
    VD rankScenario1(const Matrix &trainFeatures, const VVE &trainEntries,  const Matrix &testFeatures) {
        cerr << "=========== Rank for scenario 1 ===========" << endl;
        
        double startTime = getTime();
        
        VD res;
        VD dv;
        for (const VE &smpls : trainEntries) {
            dv.push_back(smpls[0].geniq);
        }
        
//        RRConfig conf;
//        conf.useBootstrap = true;
//        conf.bootstrapSamples = 10;
//        conf.bootstrapMode = 1;
//        conf.bootstrapIterations = 5600;
//        conf.regressionIterations = 5600;
//        conf.regressionMode = 0;
//        RidgeRegression ridge(conf);
//        ridge.train(trainFeatures.A, dv);
//        
//        // predict
//        for (int i = 0; i < testFeatures.rows(); i++) {
//            res.push_back((int)ridge.predict(testFeatures[i]));
//        }
        
        GBTConfig conf;
        conf.sampling_size_ratio = 0.5;
        conf.learning_rate = 0.001;
        conf.tree_min_nodes = 10;
        conf.tree_depth = 7;
        conf.tree_number = (int)trainFeatures.rows();
        
        GradientBoostingMachine tree(conf.sampling_size_ratio, conf.learning_rate, conf.tree_number, conf.tree_min_nodes, conf.tree_depth);
        PredictionForest *predictor = tree.train(trainFeatures.A, dv);
        
        // predict
        for (int i = 0; i < testFeatures.rows(); i++) {
            res.push_back(predictor->predict(testFeatures[i]));
        }
        
        print(res);
        
        double finishTime = getTime();
        
        Printf("Rank time: %f\n", finishTime - startTime);
        return res;
    }
};

void storeMatrixAsLibSVM(const char* fileName, const Matrix &mat, int classCol = -1) {
    FILE *fp;
    if (!(fp = fopen(fileName, "w"))) {
        throw runtime_error("Failed to open file!");
    }
    
    if (classCol < 0) {
        classCol = (int)mat.cols() - 1;
    }
    assert(classCol < mat.cols());
    // write to the buffer
    for (int row = 0; row < mat.rows(); row++) {
        // write class value first
        double val = mat(row, classCol);
        fprintf(fp, "%f", val);
        int index = 1;
        for (int col = 0; col < mat.cols(); col++) {
            if (col == classCol) {
                // skip
                continue;
            }
            val = mat(row, col);
            if (val) {
                // write only non zero
                fprintf(fp, " %d:%f", index, val);
            }
            index++;
        }
        fprintf(fp, "\n");
    }
    
    // close file
    fclose(fp);
}

#endif
