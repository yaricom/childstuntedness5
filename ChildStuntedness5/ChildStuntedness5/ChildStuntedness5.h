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
        assert(i0 >= 0 && i0 < i1 && i1 < m && j0 >= 0 && j0 < j1 && j1 < n);
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
                hasIQ = (pos == 7);
                break;
                
            case 1:
                Assert(v.size() == 16 || v.size() == 17, "Features size for second scenario, expected 16/17, but was: %lu", v.size());
                pos = parseScenario0(v);
                pos = parseScenario1(v, pos);
                hasIQ = (pos == 16);
                break;
                
            case 2:
                Assert(v.size() == 26 || v.size() == 27, "Features size for third scenario, expected 26/27, but was: %lu", v.size());
                pos = parseScenario0(v);
                pos = parseScenario1(v, pos);
                pos = parseScenario2(v, pos);
                hasIQ = (pos == 26);
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
    int gagebrth = 0, birthwt = 0, birthlen = 0, apgar1 = 0, apgar5 = 0, count = 0;
    for (const VE &smpls : data) {
        for (int i = 0; i < smpls.size(); i++) {
            gagebrth += smpls[i].gagebrth;
            birthwt += smpls[i].birthwt;
            birthlen += smpls[i].birthlen;
            apgar1 += smpls[i].apgar1;
            apgar5 += smpls[i].apgar5;
            
            count++;
        }
    }
    // find mean average
    gagebrth /= count;
    birthwt /= count;
    birthlen /= count;
    apgar1 /= count;
    apgar5 /= count;
    
    // build data matrix
    VVD features;
    for (const VE &smpls : data) {
        VD row;
        for (int i = 0; i < smpls.size(); i++) {
            row.push_back(smpls[i].sexn);
            row.push_back(smpls[i].gagebrth == Entry::NVL ? gagebrth : smpls[i].gagebrth);
            row.push_back(smpls[i].birthwt == Entry::NVL ? birthwt : smpls[i].birthwt);
            row.push_back(smpls[i].birthlen == Entry::NVL ? birthlen : smpls[i].birthlen);
            row.push_back(smpls[i].apgar1 == Entry::NVL ? apgar1 : smpls[i].apgar1);
            row.push_back(smpls[i].apgar5 == Entry::NVL ? apgar5 : smpls[i].apgar5);
            
            if (training) {
                row.push_back(smpls[i].geniq);
            }
        }
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
        
        if (scenario == 0) {
            Matrix trainMatrix = prepareScenario1Features(trainEntries, true);
            Matrix testMatrix = prepareScenario1Features(testEntries, false);
        }
        
        
        VD res;
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
