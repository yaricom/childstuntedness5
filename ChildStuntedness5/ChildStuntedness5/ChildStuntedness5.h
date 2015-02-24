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

Matrix& prepareScenario1Features(const VVE &data, const bool training) {
    // sexn, gagebrth, birthwt, birthlen, apgar1, apgar5
    int gagebrth = 0, birthwt = 0, birthlen = 0, apgar1 = 0, apgar5 = 0;
    VI counts(5, 0);
    for (const VE &smpls : data) {
        for (int i = 0; i < smpls.size(); i++) {
            if (smpls[i].gagebrth != Entry::NVL) {
                gagebrth += smpls[i].gagebrth;
                counts[0] += 1;
            }
            if (smpls[i].birthwt != Entry::NVL) {
                birthwt += smpls[i].birthwt;
                counts[1] += 1;
            }
            if (smpls[i].birthlen != Entry::NVL) {
                birthlen += smpls[i].birthlen;
                counts[2] += 1;
            }
            if (smpls[i].apgar1 != Entry::NVL) {
                apgar1 += smpls[i].apgar1;
                counts[3] += 1;
            }
            if (smpls[i].apgar5 != Entry::NVL) {
                apgar5 += smpls[i].apgar5;
                counts[4] += 1;
            }
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
    int low = pos - 1, high = pos + 1;
    while (low >= 0 && smpls[low].wtkg == Entry::NVL) low--;
    while (high < smpls.size() && smpls[high].wtkg == Entry::NVL) high++;
    double w = (smpls[pos].agedays - smpls[low].agedays) / (smpls[high].agedays - smpls[low].agedays);
    double wtkg = smpls[low].wtkg * (1 - w) + smpls[high].wtkg * w;
    return wtkg;
}

double predictL(const VE &smpls, const int pos) {
    int low = pos - 1, high = pos + 1;
    while (low >= 0 && smpls[low].htcm == Entry::NVL && smpls[low].lencm == Entry::NVL) low--;
    while (high < smpls.size() && smpls[high].htcm == Entry::NVL && smpls[high].lencm == Entry::NVL) high++;
    
    double w = (smpls[pos].agedays - smpls[low].agedays) / (smpls[high].agedays - smpls[low].agedays);
    double lowL = (smpls[low].htcm == Entry::NVL ? smpls[low].lencm : smpls[low].htcm);
    double highL = (smpls[high].htcm == Entry::NVL ? smpls[high].lencm : smpls[high].htcm);
    
    double L = lowL * (1 - w) + highL * w;
    return L;
}

Matrix& prepareScenario2Features(const VVE &data, const bool training) {
    // agedays, wtkg, htcm, lencm, bmi, waz, haz, whz, baz
    VVD features;
    for (const VE &smpls : data) {
        Printf("ID: %i, samples: %lu\n", smpls[0].subjid, smpls.size());
        
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
            wtkg += w;
            counts[0]++;
            
            // add lenght
            if (smpls[i].htcm != Entry::NVL) {
                l = smpls[i].htcm;
            } else if (smpls[i].lencm != Entry::NVL) {
                l = smpls[i].lencm;
            } else {
                l = predictL(smpls, i);
            }
            len += l;
            counts[1]++;
            
            // add bmi
            if (smpls[i].bmi != Entry::NVL) {
                bmi += smpls[i].bmi;
            } else {
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
        VD row = {wtkg / counts[0], len / counts[1], bmi / counts[2], waz / counts[3], haz / counts[4], whz / counts[5], baz / counts[6]};
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
            Matrix trainFeatures = prepareScenario1Features(trainEntries, true);
            Matrix testFeatures = prepareScenario1Features(testEntries, false);
            
            res = rankScenario1(trainFeatures, trainEntries, testFeatures, testEntries);
        } else if (scenario == 1) {
            Matrix trainFeatures0 = prepareScenario1Features(trainEntries, true);
            Matrix testFeatures0 = prepareScenario1Features(testEntries, false);
            
            Matrix trainFeatures1 = prepareScenario2Features(trainEntries, true);
            Matrix testFeatures1 = prepareScenario2Features(testEntries, false);
            
            res = rankScenario2(trainFeatures0, trainFeatures1, trainEntries, testFeatures0, testFeatures1, testEntries);
        }

        return res;
    }
    
private:
    VD rankScenario2(const Matrix &trainFeatures0, const Matrix &trainFeatures1, const VVE &trainEntries,  const Matrix &testFeatures0, const Matrix &testFeatures1, const VVE &testEntries) {
        cerr << "=========== Rank for scenario2 ===========" << endl;
        
        // predict
        VD res;
        
        return res;
    }
    
    VD rankScenario1(const Matrix &trainFeatures, const VVE &trainEntries,  const Matrix &testFeatures, const VVE &testEntries) {
        cerr << "=========== Rank for scenario 1 ===========" << endl;
        
        double startTime = getTime();
        
        RRConfig conf;
        conf.useBootstrap = true;
        conf.bootstrapSamples = 10;
        conf.bootstrapMode = 1;
        conf.bootstrapIterations = 5600;
        conf.regressionIterations = 5600;
        conf.regressionMode = 0;
        RidgeRegression ridge(conf);
        VD dv;
        for (const VE &smpls : trainEntries) {
            dv.push_back(smpls[0].geniq);
        }
        
        ridge.train(trainFeatures.A, dv);
        
        // predict
        VD res;
        int index = 0;
        for (const VE &smpls : testEntries) {
            for (int i = 0; i < smpls.size(); i++) {
                res.push_back((int)ridge.predict(testFeatures[index++]));
            }
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
