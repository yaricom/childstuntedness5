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

#define USE_RF_REGRESSION
//#define USE_REGERESSION

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
#define VE          VC < Entry >
#define VVE         VC < VE >
#define VB          VC < bool >

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

struct RNG {
    unsigned int MT[624];
    int index;
    
    RNG(int seed = 1) {
        init(seed);
    }
    
    void init(int seed = 1) {
        MT[0] = seed;
        for(int i = 1; i < 624; i++) MT[i] = (1812433253UL * (MT[i-1] ^ (MT[i-1] >> 30)) + i);
        index = 0;
    }
    
    void generate() {
        const unsigned int MULT[] = {0, 2567483615UL};
        for(int i = 0; i < 227; i++) {
            unsigned int y = (MT[i] & 0x8000000UL) + (MT[i+1] & 0x7FFFFFFFUL);
            MT[i] = MT[i+397] ^ (y >> 1);
            MT[i] ^= MULT[y&1];
        }
        for(int i = 227; i < 623; i++) {
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
    
    inline int next() {
        return rand();
    }
    
    inline int next(int x) {
        return rand() % x;
    }
    
    inline int next(int a, int b) {
        return a + (rand() % (b - a));
    }
    
    inline double nextDouble() {
        return (rand() + 0.5) * (1.0 / 4294967296.0);
    }
};
//
// ----------------------------
//
struct MT_RNG {
    typedef unsigned int uint32;
    
#define hiBit(u)       ((u) & 0x80000000U)   // mask all but highest   bit of u
#define loBit(u)       ((u) & 0x00000001U)   // mask all but lowest    bit of u
#define loBits(u)      ((u) & 0x7FFFFFFFU)   // mask     the highest   bit of u
#define mixBits(u, v)  (hiBit(u)|loBits(v))  // move hi bit of u to hi bit of v
    
    const uint32 K = 0x9908B0DFU; // a magic constant
    const uint32 N = 624; // length of state vector
    const uint32 M = 397; // a period parameter
    
    uint32   state[624 + 1];     // state vector + 1 extra to not violate ANSI C
    uint32   *next;          // next random value is computed from here
    int      left = -1;      // can *next++ this many times before reloading
    
    
    void seedMT(uint32 seed) {
        uint32 x = (seed | 1U) & 0xFFFFFFFFU, *s = state;
        int    j;
        
        for (left = 0, *s++ = x, j = N; --j;
             *s++ = (x*=69069U) & 0xFFFFFFFFU);
    }
    
    
    uint32 reloadMT(void) {
        uint32 *p0 = state, *p2 = state + 2, *pM = state + M, s0, s1;
        int    j;
        
        if (left < -1)
            seedMT(4357U);
        
        left = N - 1, next = state + 1;
        
        for (s0 = state[0], s1 = state[1], j = N - M + 1; --j; s0 = s1, s1 = *p2++)
            *p0++ = *pM++ ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? K : 0U);
        
        for (pM = state, j = M; --j; s0 = s1, s1 = *p2++)
            *p0++ = *pM++ ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? K : 0U);
        
        s1=state[0], *p0 = *pM ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? K : 0U);
        s1 ^= (s1 >> 11);
        s1 ^= (s1 <<  7) & 0x9D2C5680U;
        s1 ^= (s1 << 15) & 0xEFC60000U;
        return(s1 ^ (s1 >> 18));
    }
    
    
    uint32 randomMT() {
        uint32 y;
        
        if(--left < 0)
            return(reloadMT());
        
        y  = *next++;
        y ^= (y >> 11);
        y ^= (y <<  7) & 0x9D2C5680U;
        y ^= (y << 15) & 0xEFC60000U;
        y ^= (y >> 18);
        return(y);
    }
    
#define MAX_UINT_COKUS 4294967295  //basically 2^32-1
    double unif_rand(){
        return (((double)randomMT())/((double)MAX_UINT_COKUS));
    }
};

#define qsort_Index
void R_qsort_I(double *v, int *I, int i, int j) {
    
    int il[31], iu[31];
    double vt, vtt;
    double R = 0.375;
    int ii, ij, k, l, m;
#ifdef qsort_Index
    int it, tt;
#endif
    
    --v;
#ifdef qsort_Index
    --I;
#endif
    
    ii = i;/* save */
    m = 1;
    
L10:
    if (i < j) {
        if (R < 0.5898437) R += 0.0390625; else R -= 0.21875;
    L20:
        k = i;
        /* ij = (j + i) >> 1; midpoint */
        ij = i + (int)((j - i)*R);
#ifdef qsort_Index
        it = I[ij];
#endif
        vt = v[ij];
        if (v[i] > vt) {
#ifdef qsort_Index
            I[ij] = I[i]; I[i] = it; it = I[ij];
#endif
            v[ij] = v[i]; v[i] = vt; vt = v[ij];
        }
        /* L30:*/
        l = j;
        if (v[j] < vt) {
#ifdef qsort_Index
            I[ij] = I[j]; I[j] = it; it = I[ij];
#endif
            v[ij] = v[j]; v[j] = vt; vt = v[ij];
            if (v[i] > vt) {
#ifdef qsort_Index
                I[ij] = I[i]; I[i] = it; it = I[ij];
#endif
                v[ij] = v[i]; v[i] = vt; vt = v[ij];
            }
        }
        
        for(;;) { /*L50:*/
            //do l--;  while (v[l] > vt);
            l--;for(;v[l]>vt;l--);
            
            
#ifdef qsort_Index
            tt = I[l];
#endif
            vtt = v[l];
            /*L60:*/
            //do k++;  while (v[k] < vt);
            k=k+1;for(;v[k]<vt;k++);
            
            if (k > l) break;
            
            /* else (k <= l) : */
#ifdef qsort_Index
            I[l] = I[k]; I[k] =  tt;
#endif
            v[l] = v[k]; v[k] = vtt;
        }
        
        m++;
        if (l - i <= j - k) {
            /*L70: */
            il[m] = k;
            iu[m] = j;
            j = l;
        }
        else {
            il[m] = i;
            iu[m] = l;
            i = k;
        }
    }else { /* i >= j : */
        
    L80:
        if (m == 1)     return;
        
        /* else */
        i = il[m];
        j = iu[m];
        m--;
    }
    
    if (j - i > 10)     goto L20;
    
    if (i == ii)        goto L10;
    
    --i;
L100:
    do {
        ++i;
        if (i == j) {
            goto L80;
        }
#ifdef qsort_Index
        it = I[i + 1];
#endif
        vt = v[i + 1];
    } while (v[i] <= vt);
    
    k = i;
    
    do { /*L110:*/
#ifdef qsort_Index
        I[k + 1] = I[k];
#endif
        v[k + 1] = v[k];
        --k;
    } while (vt < v[k]);
    
#ifdef qsort_Index
    I[k + 1] = it;
#endif
    v[k + 1] = vt;
    goto L100;
}



struct RF_config {
    // number of trees in run.  200-500 gives pretty good results
    int nTree = 500;
    // number of variables to pick to split on at each node.  mdim/3 seems to give genrally good performance, but it can be altered up or down
    int mtry;
    
    // 0 or 1 (default is 1) sampling with or without replacement
    bool replace = true;
    // Minimum size of terminal nodes. Setting this number larger causes smaller trees to be grown (and thus take less time). Note that
    // the default values are different for classification (1) and regression (5).
    int nodesize = 5;
    // Should importance of predictors be assessed?
    bool importance = false;
    // Should casewise importance measure be computed? (Setting this to TRUE will override importance.)
    bool localImp = false;
    
    // Should proximity measure among the rows be calculated?
    bool proximity = false;
    // Should proximity be calculated only on 'out-of-bag' data?
    bool oob_prox = false;
    // Should an n by ntree matrix be returned that keeps track of which samples are 'in-bag' in which trees (but not how many times, if sampling with replacement)
    bool keep_inbag = false;
    // If set to TRUE, give a more verbose output as randomForest is run. If set to some integer, then running output is printed for every do_trace trees.
    int do_trace = 1;
    
    // which happens only for regression. perform bias correction for regression? Note: Experimental.risk.
    bool corr_bias = false;
    // Number of times the OOB data are permuted per tree for assessing variable
    // importance. Number larger than 1 gives slightly more stable estimate, but not
    // very effective. Currently only implemented for regression.
    int nPerm = 1;
    // a 1xD true/false vector to say which features are categorical (true), which are numeric (false)
    // maximum of 32 categories per feature is permitted
    VB categorical_feature;
    
    // whether to run run test data prediction during training against current tree
    bool testdat = false;//true;
    // the number of test trees for test data predicitions
    int nts = 10;
    // controls whether to save test set MSE labels
    bool labelts = true;
};

#define swapInt(a, b) ((a ^= b), (b ^= a), (a ^= b))

#if !defined(ARRAY_SIZE)
#define ARRAY_SIZE(x) (sizeof((x)) / sizeof((x)[0]))
#endif

/**
 * Do random forest regression.
 */
class RF_Regression {
    
    typedef enum {
        NODE_TERMINAL = -1,
        NODE_TOSPLIT  = -2,
        NODE_INTERIOR = -3
    } NodeStatus;
    
    typedef char small_int;
    
    
    // the random number generator
    MT_RNG rnd;
    
    //Global to  handle mem in findBestSplit
    int in_findBestSplit = 0; // 0 -initialize and normal.  1-normal  , -99 release
    int in_regTree = 0; //// 0 -initialize and normal.  1-normal  , -99 release
    
    //
    // the model definitions
    //
    /*  a matrix with nclass + 2 (for classification) or two (for regression) columns.
        For classification, the first nclass columns are the class-specific measures
        computed as mean decrease in accuracy. The nclass + 1st column is the
        mean decrease in accuracy over all classes. The last column is the mean decrease
        in Gini index. For Regression, the first column is the mean decrease in
        accuracy and the second the mean decrease in MSE. If importance=FALSE,
        the last measure is still returned as a vector. */
    double *impout = NULL;
    /*  The 'standard errors' of the permutation-based importance measure. For classification,
        a p by nclass + 1 matrix corresponding to the first nclass + 1
        columns of the importance matrix. For regression, a length p vector. */
    double *impSD = NULL;
    /*  a p by n matrix containing the casewise importance measures, the [i,j] element
        of which is the importance of i-th variable on the j-th case. NULL if
        localImp=FALSE. */
    double *impmat = NULL;
    // number of trees grown.
    int ntree;
    // number of predictors sampled for spliting at each node.
    int mtry;
    // the number of nodes to be created
    int nrnodes;
    // vector of mean square errors: sum of squared residuals divided by n.
    double *mse = NULL;
    // number of times cases are 'out-of-bag' (and thus used in computing OOB error estimate)
    int *nout = NULL;
    /*  if proximity=TRUE when randomForest is called, a matrix of proximity
        measures among the input (based on the frequency that pairs of data points are
        in the same terminal nodes). */
    double *prox = NULL;
    
    int *ndtree = NULL;
    small_int *nodestatus = NULL;
    int *lDaughter = NULL;
    int *rDaughter = NULL;
    double *avnode = NULL;
    int *mbest = NULL;
    double *upper = NULL;
    int *inbag = NULL;
    double *coef = NULL;
    double *y_pred_trn = NULL;
    
    // the number of categories per feature if any
    int *ncat = NULL;
    // the maximal number of categories in any feature
    int maxcat;
    // the original uniques per feature
    int **orig_uniques_in_feature = NULL;
    
public:
    
    void train(const VVD &input_X, const VD &input_Y, const RF_config &config) {
        int n_size = (int)input_X.size(); // rows
        int p_size = (int)input_X[0].size(); // cols
        
        int sampsize = n_size;
        int nodesize = config.nodesize;
        int nsum = sampsize;
        nrnodes = 2 * (int)((float)floor((float)(sampsize / ( 1 > (nodesize - 4) ? 1 : (nodesize - 4))))) + 1;
        ntree = config.nTree;
        
        Printf("sampsize: %d, nodesize: %d, nsum %d, nrnodes %d\n", sampsize, nodesize, nsum, nrnodes);
        Printf("doprox: %i, oobProx %i, biascorr %i\n", config.proximity, config.oob_prox, config.corr_bias);
        
        Assert(sampsize == input_Y.size(), "Number of samples must be equal to number of observations");
        Assert(config.mtry > 0, "Please specify number of variables to pick to split on at each node.");
        
        mtry = config.mtry;
        
        // prepare categorical inputs
        ncat = (int*) calloc(p_size, sizeof(int));
        if (config.categorical_feature.size() > 0) {
            Assert(config.categorical_feature.size() == p_size, "If provided, than list of categorical features marks must have size equal to the features dimension");
            orig_uniques_in_feature = (int **)malloc(p_size * sizeof(int *));
            for (int i = 0; i < p_size; i++) {
                if (config.categorical_feature[i]) {
                    // map categorical features
                    ncat[i] = findSortedUniqueFeaturesAndMap(input_X, i, orig_uniques_in_feature[i]);
                } else {
                    // just numerical value
                    ncat[i] = 1;
                }
            }
        } else {
            // all features numerical - set all values just to ones
            for (int i = 0; i < p_size; i++) ncat[i] = 1;
        }
        // find max of categroies
        maxcat = 1;
        for (int i = 0; i < p_size; i++) {
            maxcat = max(maxcat, ncat[i]);
        }
        
        //double y_pred_trn[n_size];
        y_pred_trn = (double*) calloc(n_size, sizeof(double));
        
        
        int imp[] = {config.importance, config.localImp, config.nPerm};
        if (imp[0] == 1) {
            impout = (double*) calloc(p_size * 2, sizeof(double));
        } else {
            impout = (double*) calloc(p_size, sizeof(double));
        }
        if (imp[1] == 1) {
            impmat = (double*) calloc(p_size * n_size, sizeof(double));
        } else {
            impmat = (double*) calloc(1, sizeof(double));
            impmat[0] = 0;
        }
        if (imp[0] == 1) {
            impSD = (double*)calloc(p_size, sizeof(double));
        } else {
            impSD = (double*)calloc(1, sizeof(double));
            impSD[0]=0;
        }
        
        // Should an n by ntree matrix be returned that keeps track of which samples are 'in-bag' in which trees (but not how many times, if sampling with replacement)
        int keepf[2];
        keepf[0] = 1;
        keepf[1] = config.keep_inbag;
        int nt;
        if (keepf[0] == 1){
            nt = ntree;
        } else {
            nt = 1;
        }
        
        // create ouput proximity matrix
        if (!config.proximity) {
            prox = (double*)calloc(1, sizeof(double));
            prox[0] = 0;
        } else {
            prox = (double*)calloc(n_size * n_size, sizeof(double));
        }
        
        //int ndtree[ntree];
        ndtree = (int*)calloc(ntree, sizeof(int));
        
        //int nodestatus[nrnodes * nt];
        nodestatus = (small_int*)calloc(nrnodes*nt, sizeof(small_int));
        
        //int lDaughter[nrnodes * nt];
        lDaughter = (int*)calloc(nrnodes*nt, sizeof(int));
        
        //int rDaughter[nrnodes * nt];
        rDaughter = (int*)calloc(nrnodes*nt, sizeof(int));
        
        //double avnode[nrnodes * nt];
        avnode = (double*) calloc(nrnodes*nt, sizeof(double));
        
        //int mbest[nrnodes * nt];
        mbest=(int*)calloc(nrnodes*nt, sizeof(int));
        
        //double upper[nrnodes * nt];
        upper = (double*) calloc(nrnodes*nt, sizeof(double));
        // vector of mean square errors: sum of squared residuals divided by n.
        mse = (double*)calloc(ntree, sizeof(double));
        
        // copy data
        double X[n_size * p_size], Y[n_size];
        int dimx[2];
        dimx[0] = n_size;
        dimx[1] = p_size;
        
        for (int i = 0; i < n_size; i++) {
            for (int j = 0; j < p_size; j++){
                if (ncat[i] == 1) {
                    // just ordinary numeric feature
                    X[i * p_size + j] = input_X[i][j];
                } else {
                    // store mapped value
                    int val = input_X[i][j];
                    for (int k = 0; k < ARRAY_SIZE(orig_uniques_in_feature[j]); k++) {
                        if (val == orig_uniques_in_feature[j][k]) {
                            val = k;
                            break;
                        }
                    }
                    X[i * p_size + j] = val;
                }
            }
            Y[i] = input_Y[i];
        }
        
        int replace = config.replace;
        int testdat = config.testdat;
        int nts = config.nts;
        
        double *xts = X;
        double *yts = Y;
        int labelts = config.labelts;
        
        //double yTestPred[nts];
        double *yTestPred; yTestPred = (double*)calloc(nts, sizeof(double));
        double proxts[] = {1};
        
        double *msets;
        if (labelts == 1) {
            msets = (double*)calloc(ntree, sizeof(double));
        } else {
            msets = (double*)calloc(ntree, sizeof(double));
            msets[0] = 1;
        }
        
        coef = (double*)calloc(2, sizeof(double));
        
        //int nout[n_size];
        nout = (int*)calloc(n_size, sizeof(int));

        if (keepf[1] == 1) {
            inbag = (int*)calloc(n_size * ntree, sizeof(int));
        } else {
            inbag = (int*)calloc(1, sizeof(int));
            inbag[0] = 1;
        }
        
        int jprint = config.do_trace;
        bool print_verbose_tree_progression = false;
        
        //below call just prints individual values
//        print_regRF_params( dimx, &sampsize,
//         &nodesize, &nrnodes, &ntree, &mtry,
//         imp, cat, config.maxcat, &jprint,
//         config.doProx, config.oobprox, config.biasCorr, y_pred_trn,
//         impout, impmat,impSD, prox,
//         ndtree, nodestatus, lDaughter, rDaughter,
//         avnode, mbest,upper, mse,
//         keepf, &replace, testdat, xts,
//         &nts, yts, labelts, yTestPred,
//         proxts, msets, coef,nout,
//         inbag);
        
        //train the RF
        regRF(X, Y, dimx, &sampsize,
              &nodesize, &nrnodes, &ntree, &mtry,
              imp, ncat, maxcat, &jprint,
              config.proximity, config.oob_prox, config.corr_bias, y_pred_trn,
              impout, impmat, impSD, prox,
              ndtree, nodestatus, lDaughter, rDaughter,
              avnode, mbest, upper, mse,
              keepf, &replace, testdat, xts,
              &nts, yts, labelts, yTestPred,
              proxts, msets, coef, nout,
              inbag, print_verbose_tree_progression) ;

        // let the train variables go free
        free(yTestPred);
        free(msets);
    }
    
    VD predict(const VVD &test_X, const RF_config &config) {
        int n_size = (int)test_X.size(); // rows
        int p_size = (int)test_X[0].size(); // cols
        double* ypred = (double*)calloc(n_size, sizeof(double));
        int mdim = p_size;
        
        double* xsplit = upper;
        double* avnodes = avnode;
        int* treeSize = ndtree;
        int keepPred = 0;
        double allPred = 0;
        int nodes = 0;
        int *nodex; nodex = (int*)calloc(n_size, sizeof(int));
        
        double* proxMat;
        if (!config.proximity) {
            proxMat = (double*)calloc(1, sizeof(double));
            proxMat[0] = 0;
        } else {
            proxMat = (double*)calloc(n_size * n_size, sizeof(double));
        }
        
        double X_test[n_size * p_size];
        for (int i = 0; i < n_size; i++) {
            for (int j = 0; j < p_size; j++){
                if (ncat[i] == 1) {
                    // just ordinary numeric feature
                    X_test[i * p_size + j] = test_X[i][j];
                } else {
                    // store mapped value
                    int val = test_X[i][j];
                    for (int k = 0; k < ARRAY_SIZE(orig_uniques_in_feature[j]); k++) {
                        if (val == orig_uniques_in_feature[j][k]) {
                            val = k;
                            break;
                        }
                    }
                    X_test[i * p_size + j] = val;
                }
            }
        }
        
        //below call just prints individual values
        //        print_regRF_params( dimx, &sampsize,
        //         &nodesize, &nrnodes, &ntree, &mtry,
        //         imp, cat, config.maxcat, &jprint,
        //         config.doProx, config.oobprox, config.biasCorr, y_pred_trn,
        //         impout, impmat,impSD, prox,
        //         ndtree, nodestatus, lDaughter, rDaughter,
        //         avnode, mbest,upper, mse,
        //         keepf, &replace, testdat, xts,
        //         &nts, yts, labelts, yTestPred,
        //         proxts, msets, coef,nout,
        //         inbag);
        
        regForest(X_test, ypred, &mdim, &n_size,
                  &ntree, lDaughter, rDaughter,
                  nodestatus, &nrnodes, xsplit,
                  avnodes, mbest, treeSize, ncat,
                  maxcat, &keepPred, &allPred, config.proximity,
                  proxMat, &nodes, nodex);
        
        VD res(n_size, 0);
        for (int i = 0;i < n_size;i++) {
            res[i] = ypred[i];
        }
        
        free(ypred);
        free(nodex);
        
        return res;
    }
    
    /**
     * Invoked to clear stored model state
     */
    void release() {
        // let the model variables go free
        free(nout);
        free(inbag);
        free(y_pred_trn);
        free(impout);
        free(impmat);
        free(impSD);
        free(mse);
        free(ndtree);
        free(nodestatus);
        free(lDaughter);
        free(rDaughter);
        free(upper);
        free(avnode);
        free(mbest);
        free(ncat);
        
        if (orig_uniques_in_feature) {
            int N = ARRAY_SIZE(orig_uniques_in_feature);
            for(int i = 0; i < N; i++) {
                free(orig_uniques_in_feature[i]);
            }
            free(orig_uniques_in_feature);
        }
    }
    
private:
    
    inline int findSortedUniqueFeaturesAndMap(const VVD input_x, const int fIndex, int *features) const {
        size_t rows = input_x.size();
        VD fTmp(rows, 0);
        for (int i = 0; i < rows; i++) {
            fTmp[i] = input_x[i][fIndex];
        }
        sort(fTmp.begin(), fTmp.end());
        VD unique;
        int previous = numeric_limits<int>::min();
        for (int i = 0; i < rows; i++) {
            if (fTmp[i] != previous) {
                previous = fTmp[i];
                unique.push_back(fTmp[i]);
            }
        }
        int catNum = (int)unique.size();
        features = (int *)malloc(catNum * sizeof(int));
        
        for (int i = 0; i < catNum; i++) {
            features[i] = (int)unique[i];
        }
        return catNum;
    }
    
    /*************************************************************************
     * Input:
     * mdim=number of variables in data set
     * nsample=number of cases
     *
     * nthsize=number of cases in a node below which the tree will not split,
     * setting nthsize=5 generally gives good results.
     *
     * nTree=number of trees in run.  200-500 gives pretty good results
     *
     * mtry=number of variables to pick to split on at each node.  mdim/3
     * seems to give genrally good performance, but it can be
     * altered up or down
     *
     * imp=1 turns on variable importance.  This is computed for the
     * mth variable as the percent rise in the test set mean sum-of-
     * squared errors when the mth variable is randomly permuted.
     *
     *************************************************************************/
    void regRF(double *x, double *y, int *xdim, int *sampsize,
               int *nthsize, int *nrnodes, int *nTree, int *mtry, int *imp,
               int *cat, int maxcat, int *jprint, int doProx, int oobprox,
               int biasCorr, double *yptr, double *errimp, double *impmat,
               double *impSD, double *prox, int *treeSize, small_int *nodestatus,
               int *lDaughter, int *rDaughter, double *avnode, int *mbest,
               double *upper, double *mse, const int *keepf, int *replace,
               int testdat, double *xts, int *nts, double *yts, int labelts,
               double *yTestPred, double *proxts, double *msets, double *coef,
               int *nout, int *inbag, int print_verbose_tree_progression) {
        
        
        double errts = 0.0, averrb, meanY, meanYts, varY, varYts, r, xrand,
        errb = 0.0, resid=0.0, ooberr, ooberrperm, delta, *resOOB;
        
        double *yb, *xtmp, *xb, *ytr, *ytree, *tgini;
        
        int k, m, mr, n, nOOB, j, jout, idx, ntest, last, ktmp, nPerm, nsample, mdim, keepF, keepInbag;
        int *oobpair, varImp, localImp, *varUsed;
        
        int *in, *nind, *nodex, *nodexts;
        
        //Abhi:temp variable
        double tmp_d;
        int tmp_i;
        small_int tmp_c;
        
        //Do initialization for COKUS's Random generator
        rnd.seedMT(2*rand()+1);  //works well with odd number so why don't use that
        
        nsample = xdim[0];
        mdim = xdim[1];
        ntest = *nts;
        varImp = imp[0];
        localImp = imp[1];
        nPerm = imp[2]; //printf("nPerm %d\n",nPerm);
        keepF = keepf[0];
        keepInbag = keepf[1];
        
        if (*jprint == 0) *jprint = *nTree + 1;
        
        yb         = (double *) calloc(*sampsize, sizeof(double));
        xb         = (double *) calloc(mdim * *sampsize, sizeof(double));
        ytr        = (double *) calloc(nsample, sizeof(double));
        xtmp       = (double *) calloc(nsample, sizeof(double));
        resOOB     = (double *) calloc(nsample, sizeof(double));
        in        = (int *) calloc(nsample, sizeof(int));
        nodex      = (int *) calloc(nsample, sizeof(int));
        varUsed    = (int *) calloc(mdim, sizeof(int));
        nind = *replace ? NULL : (int *) calloc(nsample, sizeof(int));

        oobpair = (doProx && oobprox) ?
        (int *) calloc(nsample * nsample, sizeof(int)) : NULL;
        
        /* If variable importance is requested, tgini points to the second
         "column" of errimp, otherwise it's just the same as errimp. */
        tgini = varImp ? errimp + mdim : errimp;
        
        averrb = 0.0;
        meanY = 0.0;
        varY = 0.0;
        
        zeroDouble(yptr, nsample);
        zeroInt(nout, nsample);
        for (n = 0; n < nsample; ++n) {
            varY += n * (y[n] - meanY) * (y[n] - meanY) / (n + 1);
            meanY = (n * meanY + y[n]) / (n + 1);
        }
        varY /= nsample;
        
        varYts = 0.0;
        meanYts = 0.0;
        if (testdat) {
            for (n = 0; n <= ntest; ++n) {
                varYts += n * (yts[n] - meanYts) * (yts[n] - meanYts) / (n + 1);
                meanYts = (n * meanYts + yts[n]) / (n + 1);
            }
            varYts /= ntest;
        }
        
        if (doProx) {
            zeroDouble(prox, nsample * nsample);
            if (testdat) zeroDouble(proxts, ntest * (nsample + ntest));
        }
        
        if (varImp) {
            zeroDouble(errimp, mdim * 2);
            if (localImp) zeroDouble(impmat, nsample * mdim);
        } else {
            zeroDouble(errimp, mdim);
        }
        if (labelts) zeroDouble(yTestPred, ntest);
        
        /* print header for running output */
        if (*jprint <= *nTree) {
            Printf("     |      Out-of-bag   ");
            if (testdat) Printf("|       Test set    ");
            Printf("|\n");
            Printf("Tree |      MSE  %%Var(y) ");
            if (testdat) Printf("|      MSE  %%Var(y) ");
            Printf("|\n");
        }
        /*************************************
         * Start the loop over trees.
         *************************************/
        
        time_t curr_time;
        if (testdat) {
            ytree = (double *) calloc(ntest, sizeof(double));
            nodexts = (int *) calloc(ntest, sizeof(int));
        }
        
        for (j = 0; j < *nTree; ++j) {
            
            idx = keepF ? j * *nrnodes : 0;
            zeroInt(in, nsample);
            zeroInt(varUsed, mdim);
            /* Draw a random sample for growing a tree. */
            
            if (*replace) { /* sampling with replacement */
                for (n = 0; n < *sampsize; ++n) {
                    xrand = rnd.unif_rand();
                    k = xrand * nsample;
                    in[k] = 1;
                    yb[n] = y[k];
                    for(m = 0; m < mdim; ++m) {
                        xb[m + n * mdim] = x[m + k * mdim];
                    }
                }
            } else { /* sampling w/o replacement */
                for (n = 0; n < nsample; ++n) nind[n] = n;
                last = nsample - 1;
                for (n = 0; n < *sampsize; ++n) {
                    ktmp = (int) (rnd.unif_rand() * (last+1));
                    k = nind[ktmp];
                    swapInt(nind[ktmp], nind[last]);
                    last--;
                    in[k] = 1;
                    yb[n] = y[k];
                    for(m = 0; m < mdim; ++m) {
                        xb[m + n * mdim] = x[m + k * mdim];
                    }
                }
            }
            
            if (keepInbag) {
                for (n = 0; n < nsample; ++n) inbag[n + j * nsample] = in[n];
            }
            
            /* grow the regression tree */
            regTree(xb, yb, mdim, *sampsize, lDaughter + idx, rDaughter + idx,
                    upper + idx, avnode + idx, nodestatus + idx, *nrnodes,
                    treeSize + j, *nthsize, *mtry, mbest + idx, cat, tgini,
                    varUsed);
            
            /* predict the OOB data with the current tree */
            /* ytr is the prediction on OOB data by the current tree */
            predictRegTree(x, nsample, mdim, lDaughter + idx,
                           rDaughter + idx, nodestatus + idx, ytr, upper + idx,
                           avnode + idx, mbest + idx, treeSize[j], cat, maxcat,
                           nodex);
            /* yptr is the aggregated prediction by all trees grown so far */
            errb = 0.0;
            ooberr = 0.0;
            jout = 0; /* jout is the number of cases that has been OOB so far */
            nOOB = 0; /* nOOB is the number of OOB samples for this tree */
            for (n = 0; n < nsample; ++n) {
                if (in[n] == 0) {
                    nout[n]++;
                    nOOB++;
                    yptr[n] = ((nout[n]-1) * yptr[n] + ytr[n]) / nout[n];
                    resOOB[n] = ytr[n] - y[n];
                    ooberr += resOOB[n] * resOOB[n];
                }
                if (nout[n]) {
                    jout++;
                    errb += (y[n] - yptr[n]) * (y[n] - yptr[n]);
                }
            }
            errb /= jout;
            /* Do simple linear regression of y on yhat for bias correction. */
            if (biasCorr) simpleLinReg(nsample, yptr, y, coef, &errb, nout);
            
            /* predict testset data with the current tree */
            if (testdat) {
                predictRegTree(xts, ntest, mdim, lDaughter + idx,
                               rDaughter + idx, nodestatus + idx, ytree,
                               upper + idx, avnode + idx,
                               mbest + idx, treeSize[j], cat, maxcat, nodexts);
                /* ytree is the prediction for test data by the current tree */
                /* yTestPred is the average prediction by all trees grown so far */
                errts = 0.0;
                for (n = 0; n < ntest; ++n) {
                    yTestPred[n] = (j * yTestPred[n] + ytree[n]) / (j + 1);
                }
                /* compute testset MSE */
                if (labelts) {
                    for (n = 0; n < ntest; ++n) {
                        resid = biasCorr ? yts[n] - (coef[0] + coef[1] * yTestPred[n]) : yts[n] - yTestPred[n];
                        errts += resid * resid;
                    }
                    errts /= ntest;
                }
            }
            
            /* Print running output. */
            if ((j + 1) % *jprint == 0) {
                Printf("%4d |", j + 1);
                Printf(" %8.4g %8.2f ", errb, 100 * errb / varY);
                if(labelts == 1)
                    Printf("| %8.4g %8.2f ", errts, 100.0 * errts / varYts);
                Printf("|\n");
            }
            
            mse[j] = errb;
            if (labelts) msets[j] = errts;
            
            /*  DO PROXIMITIES */
            if (doProx) {
                computeProximity(prox, oobprox, nodex, in, oobpair, nsample);
                /* proximity for test data */
                if (testdat) {
                    /* In the next call, in and oobpair are not used. */
                    computeProximity(proxts, 0, nodexts, in, oobpair, ntest);
                    for (n = 0; n < ntest; ++n) {
                        for (k = 0; k < nsample; ++k) {
                            if (nodexts[n] == nodex[k]) {
                                proxts[n + ntest * (k+ntest)] += 1.0;
                            }
                        }
                    }
                }
            }
            
            /* Variable importance */
            if (varImp) {
                for (mr = 0; mr < mdim; ++mr) {
                    if (varUsed[mr]) { /* Go ahead if the variable is used */
                        /* make a copy of the m-th variable into xtmp */
                        for (n = 0; n < nsample; ++n)
                            xtmp[n] = x[mr + n * mdim];
                        ooberrperm = 0.0;
                        for (k = 0; k < nPerm; ++k) {
                            permuteOOB(mr, x, in, nsample, mdim);
                            predictRegTree(x, nsample, mdim, lDaughter + idx,
                                           rDaughter + idx, nodestatus + idx, ytr,
                                           upper + idx, avnode + idx, mbest + idx,
                                           treeSize[j], cat, maxcat, nodex);
                            for (n = 0; n < nsample; ++n) {
                                if (in[n] == 0) {
                                    r = ytr[n] - y[n];
                                    ooberrperm += r * r;
                                    if (localImp) {
                                        impmat[mr + n * mdim] += (r * r - resOOB[n] * resOOB[n]) / nPerm;
                                    }
                                }
                            }
                        }
                        delta = (ooberrperm / nPerm - ooberr) / nOOB;
                        errimp[mr] += delta;
                        impSD[mr] += delta * delta;
                        /* copy original data back */
                        for (n = 0; n < nsample; ++n)
                            x[mr + n * mdim] = xtmp[n];
                    }
                    
                }
                
            }
        }
        /* end of tree iterations=======================================*/
        
        if (biasCorr) {  /* bias correction for predicted values */
            for (n = 0; n < nsample; ++n) {
                if (nout[n]) yptr[n] = coef[0] + coef[1] * yptr[n];
            }
            if (testdat) {
                for (n = 0; n < ntest; ++n) {
                    yTestPred[n] = coef[0] + coef[1] * yTestPred[n];
                }
            }
        }
        
        if (doProx) {
            for (n = 0; n < nsample; ++n) {
                for (k = n + 1; k < nsample; ++k) {
                    prox[nsample*k + n] /= oobprox ?
                    (oobpair[nsample*k + n] > 0 ? oobpair[nsample*k + n] : 1) :
                    *nTree;
                    prox[nsample * n + k] = prox[nsample * k + n];
                }
                prox[nsample * n + n] = 1.0;
            }
            if (testdat) {
                for (n = 0; n < ntest; ++n)
                    for (k = 0; k < ntest + nsample; ++k)
                        proxts[ntest*k + n] /= *nTree;
            }
        }
        
        if (varImp) {
            for (m = 0; m < mdim; ++m) {
                errimp[m] = errimp[m] / *nTree;
                impSD[m] = sqrt( ((impSD[m] / *nTree) -
                                  (errimp[m] * errimp[m])) / *nTree );
                if (localImp) {
                    for (n = 0; n < nsample; ++n) {
                        impmat[m + n * mdim] /= nout[n];
                    }
                }
            }
        }
        for (m = 0; m < mdim; ++m) tgini[m] /= *nTree;
        
        
        //addition by abhi
        //in order to release the space stored by the variable in findBestSplit
        // call by setting
        in_findBestSplit=-99;
        findBestSplit(&tmp_d, &tmp_i, &tmp_d, tmp_i, tmp_i,
                      tmp_i, tmp_i, &tmp_i, &tmp_d,
                      &tmp_d, &tmp_i, &tmp_i, tmp_i,
                      tmp_d, tmp_i, &tmp_i);
        
        //do the same mxFreeing of space by calling with -99
        in_regTree=-99;
        regTree(&tmp_d, &tmp_d, tmp_i, tmp_i, &tmp_i,
                &tmp_i,
                &tmp_d, &tmp_d, &tmp_c, tmp_i,
                &tmp_i, tmp_i, tmp_i, &tmp_i, &tmp_i,
                &tmp_d, &tmp_i);
        
        
        free(yb);
        free(xb);
        free(ytr);
        free(xtmp);
        free(resOOB);
        free(in);
        free(nodex);
        free(varUsed);
        if (!(*replace)  )
            free(nind);
        
        if (testdat) {
            free(ytree);
            free(nodexts);
        }
        
        if (doProx && oobprox)
            free(oobpair) ;
    }
    
    /*----------------------------------------------------------------------*/
    void regForest(double *x, double *ypred, int *mdim, int *n,
                   int *ntree, int *lDaughter, int *rDaughter,
                   small_int *nodestatus, int *nrnodes, double *xsplit,
                   double *avnodes, int *mbest, int *treeSize, int *cat,
                   int maxcat, int *keepPred, double *allpred, int doProx,
                   double *proxMat, int *nodes, int *nodex) {
        int i, j, idx1, idx2, *junk;
        double *ytree;
        
        junk = NULL;
        ytree = (double *) calloc(*n, sizeof(double));
        if (*nodes) {
            zeroInt(nodex, *n * *ntree);
        } else {
            zeroInt(nodex, *n);
        }
        if (doProx) zeroDouble(proxMat, *n * *n);
        if (*keepPred) zeroDouble(allpred, *n * *ntree);
        idx1 = 0;
        idx2 = 0;
        for (i = 0; i < *ntree; ++i) {
            zeroDouble(ytree, *n);
            predictRegTree(x, *n, *mdim, lDaughter + idx1, rDaughter + idx1,
                           nodestatus + idx1, ytree, xsplit + idx1,
                           avnodes + idx1, mbest + idx1, treeSize[i], cat, maxcat,
                           nodex + idx2);
            
            for (j = 0; j < *n; ++j) ypred[j] += ytree[j];
            if (*keepPred) {
                for (j = 0; j < *n; ++j) allpred[j + i * *n] = ytree[j];
            }
            /* if desired, do proximities for this round */
            if (doProx) computeProximity(proxMat, 0, nodex + idx2, junk,
                                         junk, *n);
            idx1 += *nrnodes; /* increment the offset */
            if (*nodes) idx2 += *n;
        }
        for (i = 0; i < *n; ++i) ypred[i] /= *ntree;
        if (doProx) {
            for (i = 0; i < *n; ++i) {
                for (j = i + 1; j < *n; ++j) {
                    proxMat[i + j * *n] /= *ntree;
                    proxMat[j + i * *n] = proxMat[i + j * *n];
                }
                proxMat[i + i * *n] = 1.0;
            }
        }
        free(ytree);
    }
    
    void simpleLinReg(int nsample, double *x, double *y, double *coef,
                      double *mse, int *hasPred) {
        /* Compute simple linear regression of y on x, returning the coefficients,
         the average squared residual, and the predicted values (overwriting y). */
        int i, nout = 0;
        double sxx=0.0, sxy=0.0, xbar=0.0, ybar=0.0;
        double dx = 0.0, dy = 0.0, py=0.0;
        
        for (i = 0; i < nsample; ++i) {
            if (hasPred[i]) {
                nout++;
                xbar += x[i];
                ybar += y[i];
            }
        }
        xbar /= nout;
        ybar /= nout;
        
        for (i = 0; i < nsample; ++i) {
            if (hasPred[i]) {
                dx = x[i] - xbar;
                dy = y[i] - ybar;
                sxx += dx * dx;
                sxy += dx * dy;
            }
        }
        coef[1] = sxy / sxx;
        coef[0] = ybar - coef[1] * xbar;
        
        *mse = 0.0;
        for (i = 0; i < nsample; ++i) {
            if (hasPred[i]) {
                py = coef[0] + coef[1] * x[i];
                dy = y[i] - py;
                *mse += dy * dy;
                /* y[i] = py; */
            }
        }
        *mse /= nout;
        return;
    }
    
    
    void regTree(double *x, double *y, int mdim, int nsample, int *lDaughter,
                 int *rDaughter,
                 double *upper, double *avnode, small_int *nodestatus, int nrnodes,
                 int *treeSize, int nthsize, int mtry, int *mbest, int *cat,
                 double *tgini, int *varUsed) {
        int i, j, k, m, ncur;
        static int *jdex, *nodestart, *nodepop;
        int ndstart, ndend, ndendl, nodecnt, jstat, msplit;
        double d, ss, av, decsplit, ubest, sumnode;
        
        if (in_regTree==-99){
            free(nodestart);
            free(jdex);
            free(nodepop);
            //      Printf("giving up mem in in_regTree\n");
            return;
        }
        
        if (in_regTree==0){
            in_regTree=1;
            nodestart = (int *) calloc(nrnodes, sizeof(int));
            nodepop   = (int *) calloc(nrnodes, sizeof(int));
            jdex = (int *) calloc(nsample, sizeof(int));
        }
        
        /* initialize some arrays for the tree */
        zeroSMALLInt(nodestatus, nrnodes);
        zeroInt(nodestart, nrnodes);
        zeroInt(nodepop, nrnodes);
        zeroDouble(avnode, nrnodes);
        
        for (i = 1; i <= nsample; ++i) jdex[i-1] = i;
        
        ncur = 0;
        nodestart[0] = 0;
        nodepop[0] = nsample;
        nodestatus[0] = NODE_TOSPLIT;
        
        /* compute mean and sum of squares for Y */
        av = 0.0;
        ss = 0.0;
        for (i = 0; i < nsample; ++i) {
            d = y[jdex[i] - 1];
            ss += i * (av - d) * (av - d) / (i + 1);
            av = (i * av + d) / (i + 1);
        }
        avnode[0] = av;
        
        /* start main loop */
        for (k = 0; k < nrnodes - 2; ++k) {
            if (k > ncur || ncur >= nrnodes - 2) break;
            /* skip if the node is not to be split */
            if (nodestatus[k] != NODE_TOSPLIT) continue;
            
            /* initialize for next call to findbestsplit */
            ndstart = nodestart[k];
            ndend = ndstart + nodepop[k] - 1;
            nodecnt = nodepop[k];
            sumnode = nodecnt * avnode[k];
            jstat = 0;
            decsplit = 0.0;
            
            findBestSplit(x, jdex, y, mdim, nsample, ndstart, ndend, &msplit,
                          &decsplit, &ubest, &ndendl, &jstat, mtry, sumnode,
                          nodecnt, cat);
            if (jstat == 1) {
                /* Node is terminal: Mark it as such and move on to the next. */
                nodestatus[k] = NODE_TERMINAL;
                continue;
            }
            /* Found the best split. */
            mbest[k] = msplit;
            varUsed[msplit - 1] = 1;
            upper[k] = ubest;
            tgini[msplit - 1] += decsplit;
            nodestatus[k] = NODE_INTERIOR;
            
            /* leftnode no.= ncur+1, rightnode no. = ncur+2. */
            nodepop[ncur + 1] = ndendl - ndstart + 1;
            nodepop[ncur + 2] = ndend - ndendl;
            nodestart[ncur + 1] = ndstart;
            nodestart[ncur + 2] = ndendl + 1;
            
            /* compute mean and sum of squares for the left daughter node */
            av = 0.0;
            ss = 0.0;
            for (j = ndstart; j <= ndendl; ++j) {
                d = y[jdex[j]-1];
                m = j - ndstart;
                ss += m * (av - d) * (av - d) / (m + 1);
                av = (m * av + d) / (m+1);
            }
            avnode[ncur+1] = av;
            nodestatus[ncur+1] = NODE_TOSPLIT;
            if (nodepop[ncur + 1] <= nthsize) {
                nodestatus[ncur + 1] = NODE_TERMINAL;
            }
            
            /* compute mean and sum of squares for the right daughter node */
            av = 0.0;
            ss = 0.0;
            for (j = ndendl + 1; j <= ndend; ++j) {
                d = y[jdex[j]-1];
                m = j - (ndendl + 1);
                ss += m * (av - d) * (av - d) / (m + 1);
                av = (m * av + d) / (m + 1);
            }
            avnode[ncur + 2] = av;
            nodestatus[ncur + 2] = NODE_TOSPLIT;
            if (nodepop[ncur + 2] <= nthsize) {
                nodestatus[ncur + 2] = NODE_TERMINAL;
            }
            
            /* map the daughter nodes */
            lDaughter[k] = ncur + 1 + 1;
            rDaughter[k] = ncur + 2 + 1;
            /* Augment the tree by two nodes. */
            ncur += 2;
        }
        *treeSize = nrnodes;
        for (k = nrnodes - 1; k >= 0; --k) {
            if (nodestatus[k] == 0) (*treeSize)--;
            if (nodestatus[k] == NODE_TOSPLIT) {
                nodestatus[k] = NODE_TERMINAL;
            }
        }
        
    }
    
    /*--------------------------------------------------------------*/
    
    void findBestSplit(double *x, int *jdex, double *y, int mdim, int nsample,
                       int ndstart, int ndend, int *msplit, double *decsplit,
                       double *ubest, int *ndendl, int *jstat, int mtry,
                       double sumnode, int nodecnt, int *cat) {
        int last, ncat[32], icat[32], lc, nl, nr, npopl, npopr;
        int i, j, kv, l;
        static int *mind, *ncase;
        static double *xt, *ut, *v, *yl;
        double sumcat[32], avcat[32], tavcat[32], ubestt;
        double crit, critmax, critvar, suml, sumr, d, critParent;
        
        
        if (in_findBestSplit==-99){
            free(ncase);
            free(mind); //had to remove this so that it wont crash for when mdim=0, strangely happened for replace=0
            free(v);
            free(yl);
            free(xt);
            free(ut);
            // Printf("giving up mem in findBestSplit\n");
            return;
        }
        
        if (in_findBestSplit==0){
            in_findBestSplit=1;
            ut = (double *) calloc(nsample, sizeof(double));
            xt = (double *) calloc(nsample, sizeof(double));
            v  = (double *) calloc(nsample, sizeof(double));
            yl = (double *) calloc(nsample, sizeof(double));
            mind  = (int *) calloc(mdim+1, sizeof(int));   //seems that the sometimes i am asking for kv[10] and that causes problesmms
            //so allocate 1 more. helps with not crashing in windows
            ncase = (int *) calloc(nsample, sizeof(int));
        }
        zeroDouble(ut, nsample);
        zeroDouble(xt, nsample);
        zeroDouble(v, nsample);
        zeroDouble(yl, nsample);
        zeroInt(mind, mdim);
        zeroInt(ncase, nsample);
        
        zeroDouble(avcat, 32);
        zeroDouble(tavcat, 32);
        
        /* START BIG LOOP */
        *msplit = -1;
        *decsplit = 0.0;
        critmax = 0.0;
        ubestt = 0.0;
        for (i=0; i < mdim; ++i) mind[i] = i;
        
        last = mdim - 1;
        for (i = 0; i < mtry; ++i) {
            critvar = 0.0;
            j = (int) (rnd.unif_rand() * (last+1));
            //Printf("j=%d, last=%d mind[j]=%d\n", j, last, mind[j]);fflush(stdout);
            kv = mind[j];
            //if(kv>100){
            //      1;
            //      getchar();
            //}
            swapInt(mind[j], mind[last]);
            /* mind[j] = mind[last];
             * mind[last] = kv; */
            last--;
            
            lc = cat[kv];
            if (lc == 1) {
                /* numeric variable */
                for (j = ndstart; j <= ndend; ++j) {
                    xt[j] = x[kv + (jdex[j] - 1) * mdim];
                    yl[j] = y[jdex[j] - 1];
                }
            } else {
                /* categorical variable */
                zeroInt(ncat, 32);
                zeroDouble(sumcat, 32);
                for (j = ndstart; j <= ndend; ++j) {
                    l = (int) x[kv + (jdex[j] - 1) * mdim];
                    sumcat[l - 1] += y[jdex[j] - 1];
                    ncat[l - 1] ++;
                }
                /* Compute means of Y by category. */
                for (j = 0; j < lc; ++j) {
                    avcat[j] = ncat[j] ? sumcat[j] / ncat[j] : 0.0;
                }
                /* Make the category mean the `pseudo' X data. */
                for (j = 0; j < nsample; ++j) {
                    xt[j] = avcat[(int) x[kv + (jdex[j] - 1) * mdim] - 1];
                    yl[j] = y[jdex[j] - 1];
                }
            }
            /* copy the x data in this node. */
            for (j = ndstart; j <= ndend; ++j) v[j] = xt[j];
            for (j = 1; j <= nsample; ++j) ncase[j - 1] = j;
            R_qsort_I(v, ncase, ndstart + 1, ndend + 1);
            if (v[ndstart] >= v[ndend]) continue;
            /* ncase(n)=case number of v nth from bottom */
            /* Start from the right and search to the left. */
            critParent = sumnode * sumnode / nodecnt;
            suml = 0.0;
            sumr = sumnode;
            npopl = 0;
            npopr = nodecnt;
            crit = 0.0;
            /* Search through the "gaps" in the x-variable. */
            for (j = ndstart; j <= ndend - 1; ++j) {
                d = yl[ncase[j] - 1];
                suml += d;
                sumr -= d;
                npopl++;
                npopr--;
                if (v[j] < v[j+1]) {
                    crit = (suml * suml / npopl) + (sumr * sumr / npopr) -
                    critParent;
                    if (crit > critvar) {
                        ubestt = (v[j] + v[j+1]) / 2.0;
                        critvar = crit;
                    }
                }
            }
            if (critvar > critmax) {
                *ubest = ubestt;
                *msplit = kv + 1;
                critmax = critvar;
                for (j = ndstart; j <= ndend; ++j) {
                    ut[j] = xt[j];
                }
                if (cat[kv] > 1) {
                    for (j = 0; j < cat[kv]; ++j) tavcat[j] = avcat[j];
                }
            }
        }
        *decsplit = critmax;
        
        /* If best split can not be found, set to terminal node and return. */
        if (*msplit != -1) {
            nl = ndstart;
            for (j = ndstart; j <= ndend; ++j) {
                if (ut[j] <= *ubest) {
                    nl++;
                    ncase[nl-1] = jdex[j];
                }
            }
            *ndendl = imax2(nl - 1, ndstart);
            nr = *ndendl + 1;
            for (j = ndstart; j <= ndend; ++j) {
                if (ut[j] > *ubest) {
                    if (nr >= nsample) break;
                    nr++;
                    ncase[nr - 1] = jdex[j];
                }
            }
            if (*ndendl >= ndend) *ndendl = ndend - 1;
            for (j = ndstart; j <= ndend; ++j) jdex[j] = ncase[j];
            
            lc = cat[*msplit - 1];
            if (lc > 1) {
                for (j = 0; j < lc; ++j) {
                    icat[j] = (tavcat[j] < *ubest) ? 1 : 0;
                }
                *ubest = pack(lc, icat);
            }
        } else *jstat = 1;
        
    }
    /*====================================================================*/
    void predictRegTree(double *x, int nsample, int mdim,
                        int *lDaughter, int *rDaughter, small_int *nodestatus,
                        double *ypred, double *split, double *nodepred,
                        int *splitVar, int treeSize, int *cat, int maxcat,
                        int *nodex) {
        int i, j, k, m, *cbestsplit = NULL;
        unsigned int npack;
        
        /* decode the categorical splits */
        if (maxcat > 1) {
            cbestsplit = (int *) calloc(maxcat * treeSize, sizeof(int));
            zeroInt(cbestsplit, maxcat * treeSize);
            for (i = 0; i < treeSize; ++i) {
                if (nodestatus[i] != NODE_TERMINAL && cat[splitVar[i] - 1] > 1) {
                    npack = (unsigned int) split[i];
                    /* unpack `npack' into bits */
                    for (j = 0; npack; npack >>= 1, ++j) {
                        cbestsplit[j + i*maxcat] = npack & 1;
                    }
                }
            }
        }
        
        for (i = 0; i < nsample; ++i) {
            k = 0;
            while (nodestatus[k] != NODE_TERMINAL) { /* go down the tree */
                m = splitVar[k] - 1;
                if (cat[m] == 1) {
                    k = (x[m + i*mdim] <= split[k]) ?
                    lDaughter[k] - 1 : rDaughter[k] - 1;
                } else if (cbestsplit){
                    /* Split by a categorical predictor */
                    k = cbestsplit[(int) x[m + i * mdim] - 1 + k * maxcat] ?
                    lDaughter[k] - 1 : rDaughter[k] - 1;
                }
            }
            /* terminal node: assign prediction and move on to next */
            ypred[i] = nodepred[k];
            nodex[i] = k + 1;
        }
        if (maxcat > 1) free(cbestsplit);
    }
    
    void zeroSMALLInt(void *x, int length) {
        memset(x, 0, length * sizeof(small_int));
    }
    void zeroInt(int *x, int length) {
        memset(x, 0, length * sizeof(int));
    }
    
    void zeroDouble(double *x, int length) {
        memset(x, 0, length * sizeof(double));
    }
    
    int imax2(int x, int y) {
        return (x < y) ? y : x;
    }
    
    
    int pack(int nBits, int *bits) {
        int i = nBits, pack = 0;
        while (--i >= 0) pack += bits[i] << i;
        return(pack);
    }
    
    void unpack(unsigned int pack, int *bits) {
        /* pack is a 4-byte integer.  The sub. returns icat, an integer array of
         zeroes and ones corresponding to the coefficients in the binary expansion
         of pack. */
        int i;
        for (i = 0; pack != 0; pack >>= 1, ++i) bits[i] = pack & 1;
    }
    
    /* Compute proximity. */
    void computeProximity(double *prox, int oobprox, int *node, int *inbag,
                          int *oobpair, int n) {
        /* Accumulate the number of times a pair of points fall in the same node.
         prox:    n x n proximity matrix
         oobprox: should the accumulation only count OOB cases? (0=no, 1=yes)
         node:    vector of terminal node labels
         inbag:   indicator of whether a case is in-bag
         oobpair: matrix to accumulate the number of times a pair is OOB together
         n:       total number of cases
         */
        int i, j;
        for (i = 0; i < n; ++i) {
            for (j = i+1; j < n; ++j) {
                if (oobprox) {
                    if ((inbag[i] > 0) ^ (inbag[j] > 0)) {
                        oobpair[j*n + i] ++;
                        oobpair[i*n + j] ++;
                        if (node[i] == node[j]) {
                            prox[j*n + i] += 1.0;
                            prox[i*n + j] += 1.0;
                        }
                    }
                } else {
                    if (node[i] == node[j]) {
                        prox[j*n + i] += 1.0;
                        prox[i*n + j] += 1.0;
                    }
                }
            }
        }
    }
    
    void permuteOOB(int m, double *x, int *in, int nsample, int mdim) {
        /* Permute the OOB part of a variable in x.
         * Argument:
         *   m: the variable to be permuted
         *   x: the data matrix (variables in rows)
         *   in: vector indicating which case is OOB
         *   nsample: number of cases in the data
         *   mdim: number of variables in the data
         */
        double *tp, tmp;
        int i, last, k, nOOB = 0;
        
        tp = (double *) calloc(nsample , sizeof(double));
        
        for (i = 0; i < nsample; ++i) {
            /* make a copy of the OOB part of the data into tp (for permuting) */
            if (in[i] == 0) {
                tp[nOOB] = x[m + i*mdim];
                nOOB++;
            }
        }
        /* Permute tp */
        last = nOOB;
        for (i = 0; i < nOOB; ++i) {
            k = (int) (last * rnd.unif_rand());
            tmp = tp[last - 1];
            tp[last - 1] = tp[k];
            tp[k] = tmp;
            last--;
        }
        
        /* Copy the permuted OOB data back into x. */
        nOOB = 0;
        for (i = 0; i < nsample; ++i) {
            if (in[i] == 0) {
                x[m + i*mdim] = tp[nOOB];
                nOOB++;
            }
        }
        free(tp);
    }
    
    void print_regRF_params( int *xdim, int *sampsize,
                            int *nthsize, int *nrnodes, int *nTree, int *mtry, int *imp,
                            int *cat, int maxcat, int *jprint, int doProx, int oobprox,
                            int biasCorr, double *yptr, double *errimp, double *impmat,
                            double *impSD, double *prox, int *treeSize, small_int *nodestatus,
                            int *lDaughter, int *rDaughter, double *avnode, int *mbest,
                            double *upper, double *mse, int *keepf, int *replace,
                            int testdat, double *xts, int *nts, double *yts, int labelts,
                            double *yTestPred, double *proxts, double *msets, double *coef,
                            int *nout, int *inbag)  {
        Printf("n_size %d p_size %d\n", xdim[0], xdim[1]);
        Printf("sampsize %d, nodesize %d nrnodes %d\n", *sampsize, *nthsize, *nrnodes);
        Printf("ntree %d, mtry %d, impor %d, localimp %d, nPerm %d\n", *nTree, *mtry, imp[0], imp[1], imp[2]);
        Printf("maxcat %d, jprint %d, doProx %d, oobProx %d, biasCorr %d\n", maxcat, *jprint, doProx, oobprox, biasCorr);
        Printf("prox %f, keep.forest %d, keep.inbag %d\n", *prox, keepf[0], keepf[1]);
        Printf("replace %d, labelts %d, proxts %f\n", *replace, labelts, *proxts);
    }
};

//
// ----------------------------
//

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
        std::advance(r, rand() % left);
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
    void buildRegressionTree(const VC<VD> &samples_x, const VD &obs_y) {
        size_t samples_num = samples_x.size();
        
        Assert(samples_num == obs_y.size() && samples_num != 0,
               "The number of samles does not match with the number of observations or the samples number is 0. Samples: %i", samples_num);
        
        Assert (m_min_nodes * 2 <= samples_num, "The number of samples is too small");
        
        size_t feature_dim = samples_x[0].size();
        features_importance.resize(feature_dim, 0);
        
        // build the regression tree
        buildTree(samples_x, obs_y);
    }
    
private:
    
    /*
     *  The following function gets the best split given the data
     */
    BestSplit findOptimalSplit(const VC<VD> &samples_x, const VD &obs_y) {
        
        BestSplit split_point;
        
        if (m_current_depth > m_max_depth) {
            return split_point;
        }
        
        size_t samples_num = samples_x.size();
        
        if (m_min_nodes * 2 > samples_num) {
            // the number of observations in terminals is too small
            return split_point;
        }
        size_t feature_dim = samples_x[0].size();
        
        
        double min_err = 0;
        int split_index = -1;
        double node_value = 0.0;
        
        // begin to get the best split information
        for (int loop_i = 0; loop_i < feature_dim; loop_i++){
            // get the optimal split for the loop_index feature
            
            // get data sorted by the loop_i-th feature
            VC<ListData> list_by_feature;
            for (int loop_j = 0; loop_j < samples_num; loop_j++) {
                list_by_feature.push_back(ListData(samples_x[loop_j][loop_i], obs_y[loop_j]));
            }
            
            // sort the list by feature value ascending
            sort(list_by_feature.begin(), list_by_feature.end());
            
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
                ListData fetched_data = list_by_feature[loop_j];
                sum_left += fetched_data.m_y;
                count_left++;
            }
            mean_left = sum_left / count_left;
            
            // initialize right
            for (int loop_j = m_min_nodes; loop_j < samples_num; loop_j++) {
                ListData fetched_data = list_by_feature[loop_j];
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
            current_node_value = (list_by_feature[m_min_nodes].m_x + list_by_feature[m_min_nodes - 1].m_x) / 2;
            
            if (current_err < min_err && current_node_value != list_by_feature[m_min_nodes - 1].m_x) {
                split_index = loop_i;
                node_value = current_node_value;
                min_err = current_err;
            }
            
            // begin to find the best split point for the feature
            for (int loop_j = m_min_nodes; loop_j < samples_num - m_min_nodes; loop_j++) {
                ListData fetched_data = list_by_feature[loop_j];
                double y = fetched_data.m_y;
                sum_left += y;
                count_left++;
                mean_left = sum_left / count_left;
                
                
                sum_right -= y;
                count_right--;
                mean_right = sum_right / count_right;
                
                
                current_err = -1 * count_left * mean_left * mean_left - count_right * mean_right * mean_right;
                // current node value
                current_node_value = (list_by_feature[loop_j + 1].m_x + fetched_data.m_x) / 2;
                
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
    SplitRes splitData(const VC<VD> &samples_x, const VD &obs_y, const BestSplit &best_split) {
        
        SplitRes split_res;
        
        int feature_index = best_split.m_feature_index;
        double node_value = best_split.m_node_value;
        
        size_t samples_count = obs_y.size();
        for (int loop_i = 0; loop_i < samples_count; loop_i++) {
            VD ith_sample = samples_x[loop_i];
            if (ith_sample[feature_index] < node_value) {
                split_res.m_feature_left.push_back(ith_sample);
                split_res.m_obs_left.push_back(obs_y[loop_i]);
            } else {
                split_res.m_feature_right.push_back(ith_sample);
                split_res.m_obs_right.push_back(obs_y[loop_i]);
            }
        }
        
        // update terminal values
        if (m_type == AVERAGE) {
            double mean_value = 0.0;
            for (const double obsL : split_res.m_obs_left) {
                mean_value += obsL;
            }
            mean_value = mean_value / split_res.m_obs_left.size();
            split_res.m_left_value = mean_value;
            
            mean_value = 0.0;
            for (const double obsR : split_res.m_obs_right) {
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
    Node* buildTree(const VC<VD> &samples_x, const VD &obs_y) {
        
        // obtain the optimal split point
        m_current_depth = m_current_depth + 1;
        
        BestSplit best_split = findOptimalSplit(samples_x, obs_y);
        
        if (!best_split.m_status) {
            if (m_current_depth > 0)
                m_current_depth = m_current_depth - 1;
            
            return NULL;
        }
        
        // update feature importance info
        features_importance[best_split.m_feature_index] += 1;
        
        // split the data
        SplitRes split_data = splitData(samples_x, obs_y, best_split);
        
        // append current value to tree
        Node *new_node = new Node(best_split.m_node_value, best_split.m_feature_index,
                                  split_data.m_left_value, split_data.m_right_value);
        
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
            for (const double d : input_y) {
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
                
                for (const int sel_index : sampled_index) {
                    // assign value
                    train_y.push_back(gradient[sel_index]);
                    train_x.push_back(input_x[sel_index]);
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
    
    Matrix& subMatrix(const int i0, const int i1, const VI &c) {
        assert(i0 >= 0 && i0 <= i1 && i1 < m);
        Matrix *X = new Matrix(i1 - i0 + 1, c.size());
        for (int i = i0; i <= i1; i++) {
            for (int j = 0; j < c.size(); j++) {
                assert(c[j] < n && c[j] >= 0);
                X->A[i - i0][j] = A[i][c[j]];
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

VD expandRow(const VD &row) {
    size_t N = row.size();
    
    VD rv, rvM, rvD;
    for (int i = 0; i < N; i++) {
        rv.push_back(row[i]);
        
        for (int j = 0; j < i; j++) {
            // find multiplication
            rvM.push_back(row[i] * row[j]);
            // find division
            double v = row[j] ? row[i] / row[j] : 0;
            rvD.push_back(v);
        }
    }
    
    // append vectors
    rv.reserve(rv.size() + rvM.size() + rvD.size());
    rv.insert(rv.end(), rvM.begin(), rvM.end());
    rv.insert(rv.end(), rvD.begin(), rvD.end());
    
    return rv;
}

Matrix& expandMatrixLineary(const Matrix &mat) {
    VVD rv;
    for (int i = 0; i < mat.rows(); i++) {
        rv.push_back(expandRow(mat[i]));
    }
    
    Matrix *ret = new Matrix(rv);
    return *ret;
}

//
// -----------------------------------------
//

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
        subjid = (int)stof(v[pos++]);
        agedays = (int)stof(v[pos++]);
        wtkg = stof(v[pos++]);
        htcm = stof(v[pos++]);
        lencm = stof(v[pos++]);
        bmi = stof(v[pos++]);
        waz = stof(v[pos++]);
        haz = stof(v[pos++]);
        whz = stof(v[pos++]);
        baz = stof(v[pos++]);
        siteid = (int)stof(v[pos++]);
        sexn = (int)stof(v[pos++]);
        feedingn = (int)stof(v[pos++]);
        gagebrth = (int)stof(v[pos++]);
        birthwt = (int)stof(v[pos++]);
        birthlen = (int)stof(v[pos++]);
        apgar1 = (int)stof(v[pos++]);
        apgar5 = (int)stof(v[pos++]);
        mage = (int)stof(v[pos++]);
        demo1n = (int)stof(v[pos++]);
        mmaritn = (int)stof(v[pos++]);
        mcignum = (int)stof(v[pos++]);
        parity = (int)stof(v[pos++]);
        gravida = (int)stof(v[pos++]);
        meducyrs = (int)stof(v[pos++]);
        demo2n = (int)stof(v[pos++]);
        
        // store IQ
        if (v.size() == 27) {
            geniq = (int)stof(v[pos]);
        }
        
        // correct missing values
        if (agedays == 1) {
            if (wtkg == NVL && birthwt != NVL) {
                wtkg = birthwt / 1000.0;
            }
            if (lencm == NVL && birthlen != NVL) {
                lencm = birthlen;
            }
        }
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

Matrix& prepareScenario1Features(VVE &data) {
    // sexn, gagebrth, birthwt, birthlen, apgar1, apgar5
    int gagebrth = 0;
    VI counts(5, 0);
    VVI birthwt(2, VI(2, 0));
    VVI birthlen(2, VI(2, 0));
    VVI apgar1, apgar5;
    int idx = 0;
    for (const VE &smpls : data) {
        if (smpls[0].gagebrth != Entry::NVL) {
            gagebrth += smpls[0].gagebrth;
            counts[0]++;
        }
        if (smpls[0].birthwt != Entry::NVL) {
            birthwt[smpls[0].sexn - 1][0] += smpls[0].birthwt;
            birthwt[smpls[0].sexn - 1][1]++;
        }
        if (smpls[0].birthlen != Entry::NVL) {
            birthlen[smpls[0].sexn - 1][0] += smpls[0].birthlen;
            birthlen[smpls[0].sexn - 1][1]++;
        }
        idx = smpls[0].sexn * smpls[0].gagebrth;
        if (smpls[0].apgar1 != Entry::NVL) {
            if (idx >= apgar1.size()) {
                apgar1.resize(idx + 1, VI(2, 0));
            }
            apgar1[idx][0] += smpls[0].apgar1;
            apgar1[idx][1]++;
        }
        if (smpls[0].apgar5 != Entry::NVL) {
            if (idx >= apgar5.size()) {
                apgar5.resize(idx + 1, VI(2, 0));
            }
            apgar5[idx][0] += smpls[0].apgar5;
            apgar5[idx][1]++;
        }
    }
    // find mean average
    gagebrth    /= counts[0];
    
    RNG rnd;
    // build data matrix
    VVD features;
    for (VE &smpls : data) {
        VD row;
        row.push_back(smpls[0].sexn);
        
        row.push_back(smpls[0].gagebrth == Entry::NVL ? gagebrth : smpls[0].gagebrth);
        if (smpls[0].birthwt == Entry::NVL) {
            smpls[0].birthwt = birthwt[smpls[0].sexn - 1][0] / birthwt[smpls[0].sexn - 1][1];
        }
        row.push_back(smpls[0].birthwt);
        
        if (smpls[0].birthlen == Entry::NVL) {
            smpls[0].birthlen = birthlen[smpls[0].sexn - 1][0] / birthlen[smpls[0].sexn - 1][1];
        }
        row.push_back(smpls[0].birthlen);
        
        row.push_back(smpls[0].apgar1 == Entry::NVL ? rnd.next() % 11 : smpls[0].apgar1);
        row.push_back(smpls[0].apgar5 == Entry::NVL ? rnd.next() % 11 : smpls[0].apgar5);
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
            } else if (smpls[i].agedays == 1 && smpls[i].birthwt != Entry::NVL) {
                w = smpls[i].birthwt;
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
            } else if (smpls[i].agedays == 1 && smpls[i].birthlen != Entry::NVL) {
                l = smpls[i].birthlen;
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
        VD row = {safeDiv(wtkg, counts[0], Entry::NVL), safeDiv(len, counts[1], Entry::NVL), safeDiv(bmi, counts[2], Entry::NVL), safeDiv(waz, counts[3], Entry::NVL), safeDiv(haz, counts[4], Entry::NVL), safeDiv(whz, counts[5], Entry::NVL), safeDiv(baz, counts[6], Entry::NVL)};
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
    
    RNG rnd;
    // build data matrix
    VVD features;
    for (const VE &smpls : data) {
        VD row;
        row.push_back(smpls[0].siteid);
        row.push_back(smpls[0].feedingn != Entry::NVL ? smpls[0].feedingn : rnd.next(3) + 1 );
        row.push_back(smpls[0].mage != Entry::NVL ? smpls[0].mage : mage);
        row.push_back(smpls[0].demo1n != Entry::NVL ? smpls[0].demo1n : rnd.next(2) + 1);
        row.push_back(smpls[0].mmaritn != Entry::NVL ? smpls[0].mmaritn : rnd.next(6) + 1);
        row.push_back(smpls[0].mcignum != Entry::NVL ? smpls[0].mcignum : 0);
        row.push_back(smpls[0].parity != Entry::NVL ? smpls[0].parity : parity);
        row.push_back(smpls[0].gravida != Entry::NVL ? smpls[0].gravida : gravida);
        row.push_back(smpls[0].meducyrs != Entry::NVL ? smpls[0].meducyrs : meducyrs);
        row.push_back(smpls[0].demo2n != Entry::NVL ? smpls[0].demo2n : rnd.next(5) + 1);
        
        features.push_back(row);
    }
    Matrix *m = new Matrix(features);
    return *m;
}

#ifdef LOCAL
void storeMatrixAsLibSVM(const char* fileName, const Matrix &mat, const VD &dv);
#endif

class ChildStuntedness5 {
    
public:
    /**
     * @param testType The testType parameter will be 0, 1, or 2, to indicate Example, Provisional, or System test
     * @param scenario The scenario parameter is also 0, 1, or 2, referring to the three scenarios listed above.
     */
    VD predict(const int testType, const int scenario, const VS &training, const VS &testing) {
        size_t X = training.size();
        size_t Y = testing.size();
        
        cerr << "---------------------------------------------" << endl;
        fprintf(stderr, "Test type: %i, scenario: %i, training size: %lu, test size: %lu\n", testType, scenario, X, Y);
        
        VVE trainEntries = readEntries(training, scenario);
        VVE testEntries = readEntries(testing, scenario);
        
        fprintf(stderr, "Training subjects: %lu, test subjects: %lu\n", trainEntries.size(), testEntries.size());
        
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
#ifdef USE_REGERESSION
    VD rankScenario3(const Matrix &trainFeatures0, const Matrix &trainFeatures1, const Matrix &trainFeatures2, const VVE &trainEntries,
                     const Matrix &testFeatures0, const Matrix &testFeatures1, const Matrix &testFeatures2) {
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
        conf.tree_number = 2500;
        
        
        GradientBoostingMachine tree(conf.sampling_size_ratio, conf.learning_rate, conf.tree_number, conf.tree_min_nodes, conf.tree_depth);
        PredictionForest *predictor = tree.train(trainFeatures.A, dv);
        
        // predict
        for (int i = 0; i < testFeatures.rows(); i++) {
            res.push_back(predictor->predict(testFeatures[i]));
        }
        
        //        print(res);
        
        double finishTime = getTime();
        
        Printf("Rank time: %f\n", finishTime - startTime);
        
        return res;
    }
    
    VD rankScenario2(const Matrix &trainFeatures0, const Matrix &trainFeatures1, const VVE &trainEntries,  const Matrix &testFeatures0, const Matrix &testFeatures1) {
        cerr << "=========== Rank for scenario 1 ===========" << endl;
        
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
        conf.tree_number = 3000;//1500;
        
        GradientBoostingMachine tree(conf.sampling_size_ratio, conf.learning_rate, conf.tree_number, conf.tree_min_nodes, conf.tree_depth);
        PredictionForest *predictor = tree.train(trainFeatures.A, dv);
        
        // predict
        for (int i = 0; i < testFeatures.rows(); i++) {
            res.push_back(predictor->predict(testFeatures[i]));
        }
        
        //        print(res);
        
        double finishTime = getTime();
        
        Printf("Rank time: %f\n", finishTime - startTime);
        
        return res;
    }
    
    VD rankScenario1(const Matrix &trainFeatures, const VVE &trainEntries,  const Matrix &testFeatures) {
        cerr << "=========== Rank for scenario 0 ===========" << endl;
        
        double startTime = getTime();
        
        VD res;
        VD dv;
        for (const VE &smpls : trainEntries) {
            dv.push_back(smpls[0].geniq);
        }
        
        GBTConfig conf;
        conf.sampling_size_ratio = 0.5;
        conf.learning_rate = 0.001;
        conf.tree_min_nodes = 10;
        conf.tree_depth = 7;
        conf.tree_number = 3000;//2100;//
        
        GradientBoostingMachine tree(conf.sampling_size_ratio, conf.learning_rate, conf.tree_number, conf.tree_min_nodes, conf.tree_depth);
        PredictionForest *predictor = tree.train(trainFeatures.A, dv);
        
        // predict
        for (int i = 0; i < testFeatures.rows(); i++) {
            res.push_back(predictor->predict(testFeatures[i]));
        }
        
        //        print(res);
        
        double finishTime = getTime();
        
        Printf("Rank time: %f\n", finishTime - startTime);
        return res;
    }
#endif
    
#ifdef USE_RF_REGRESSION
    VD rankScenario3(const Matrix &trainFeatures0, const Matrix &trainFeatures1, const Matrix &trainFeatures2, const VVE &trainEntries,
                     const Matrix &testFeatures0, const Matrix &testFeatures1, const Matrix &testFeatures2) {
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
        trainFeatures.concat(trainFeatures2);
        
        Matrix testFeatures = testFeatures0;
        testFeatures.concat(testFeatures1);
        testFeatures.concat(testFeatures2);
        
        
        RF_config conf;
        conf.nTree = 1000;
        conf.mtry = 10;
        
        RF_Regression rfRegr;
        rfRegr.train(trainFeatures.A, dv, conf);
        VD res = rfRegr.predict(testFeatures.A, conf);
        
        rfRegr.release();
        
        //        print(res);
        
        double finishTime = getTime();
        
        Printf("Rank time: %f\n", finishTime - startTime);
        
        return res;
    }
    
    VD rankScenario2(const Matrix &trainFeatures0, const Matrix &trainFeatures1, const VVE &trainEntries,  const Matrix &testFeatures0, const Matrix &testFeatures1) {
        cerr << "=========== Rank for scenario 1 ===========" << endl;
        
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
        
        
        RF_config conf;
        conf.nTree = 1500;
        conf.mtry = 1;
        
        RF_Regression rfRegr;
        rfRegr.train(trainFeatures.A, dv, conf);
        VD res = rfRegr.predict(testFeatures.A, conf);
        
        rfRegr.release();
        
        //        print(res);
        
        double finishTime = getTime();
        
        Printf("Rank time: %f\n", finishTime - startTime);
        
        return res;
    }
    
    VD rankScenario1(const Matrix &trainFeatures, const VVE &trainEntries,  const Matrix &testFeatures) {
        cerr << "=========== Rank for scenario 0 ===========" << endl;
        
        double startTime = getTime();
        
        VD dv;
        for (const VE &smpls : trainEntries) {
            dv.push_back(smpls[0].geniq);
        }
        
        RF_config conf;
        conf.nTree = 500;
        conf.mtry = 1;//6;
        conf.testdat = true;
        
        
        RF_Regression rfRegr;
        rfRegr.train(trainFeatures.A, dv, conf);
        VD res = rfRegr.predict(testFeatures.A, conf);
        
        rfRegr.release();
        
        //        print(res);
        
        double finishTime = getTime();
        
        Printf("Rank time: %f\n", finishTime - startTime);
        return res;
    }
    
#endif

};


void storeMatrixAsLibSVM(const char* fileName, const Matrix &mat, const VD &dv) {
    FILE *fp;
    if (!(fp = fopen(fileName, "w"))) {
        throw runtime_error("Failed to open file!");
    }
    Assert(mat.rows() == dv.size(), "Number of rows in matrix must be equal to size of DV");
    
    // write to the buffer
    for (int row = 0; row < mat.rows(); row++) {
        // write class value first
        double val = dv[row];
        fprintf(fp, "%f", val);
        int index = 1;
        for (int col = 0; col < mat.cols(); col++) {
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
