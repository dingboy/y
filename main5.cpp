// 单线程生成向量
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include <pthread.h>
#include <vector>
#include <map>

using namespace std;

const double pi = 3.141592653589793238462643383;

int transeThreads = 8;
int transeTrainTimes = 1000;
int nbatches = 1;
const int dimension = 100;
double transeAlpha = 0.001;
double margin = 1.0;
double margin2 = 1.0;
double regular_c = 1.0;
double beta = 0.001;
double margin_rate = 10.0;
double margin_e = 0.1;

string inPath = "../data/";
string outPath = "./outvec/";
string version = "y_c_3_gen_p_0.01";

int *lefHead, *rigHead;
int *lefTail, *rigTail;

struct Triple {
	int h, r, t;
	bool operator <(const Triple& other) const {
		return (r < other.r)||(r == other.r && h < other.h)||(h == other.h && r == other.r && t < other.t);
	}
};
struct Relation_Info {
	int maxh, maxt;
	double meanh, meant;
};

Triple *trainHead, *trainTail, *trainList;

int relationTotal, entityTotal, tripleTotal;
double *relationVec, *entityVec;
typedef double (*array_vec)[dimension];
vector<vector<vector<double> > > entv;
vector<vector<double> > relv;
vector<Relation_Info> num_of_rel;
vector<map<Triple, int> > info_of_rel;
vector<vector<double> > factor; //mixture factor
vector<vector<int> > count_for_factor;
vector<vector<double> > sigma; // variance of an enity
int bad_triple, new_vector_num;

struct cmp_head {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

struct cmp_tail {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
};

/*
	There are some math functions for the program initialization.
*/
unsigned long long *next_random;

unsigned long long randd(int id) {
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

int rand_max(int id, int x) {
	int res = randd(id) % x;
	while (res<0)
		res+=x;
	return res;
}

double rand(double min, double max) {
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

double normal(double x, double miu,double sigma) {
	return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

double randn(double miu,double sigma, double min ,double max) {
	double x, y, dScope;
	do {
		x = rand(min,max);
		y = normal(x,miu,sigma);
		dScope=rand(0.0,normal(miu,miu,sigma));
	} while (dScope > y);
	return x;
}

void norm(double * con) {
	double x = 0;
	for (int  ii = 0; ii < dimension; ii++)
		x += (*(con + ii)) * (*(con + ii));
	x = sqrt(x);
	if (x>1)
		for (int ii=0; ii < dimension; ii++)
			*(con + ii) /= x;
}

/*
	Read triples from the training file.
*/

void init() {

	FILE *fin;
	int tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relationTotal);
	fclose(fin);

	relationVec = (double *)calloc(relationTotal * dimension, sizeof(double));
	for (int i = 0; i < relationTotal; i++) {
		for (int ii=0; ii<dimension; ii++)
			relationVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
	}

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entityTotal);
	fclose(fin);

	entityVec = (double *)calloc(entityTotal * dimension, sizeof(double));
	for (int i = 0; i < entityTotal; i++) {
		for (int ii=0; ii<dimension; ii++)
			entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
		// norm(entityVec+i*dimension);
	}

	relv.resize(relationTotal);
	for (int i = 0; i < relationTotal; i++) {
		relv[i].resize(dimension);
		for (int ii=0; ii<dimension; ii++)
			relv[i][ii] = randn(0, 1.0 / dimension, -6.0 / sqrt(dimension), 6.0 / sqrt(dimension));
	}
	entv.resize(entityTotal);
	for (int i = 0; i < entityTotal; i++) {
		entv[i].reserve(4);
		vector<double> temp;
		temp.resize(100);
		for (int iii=0; iii<dimension; iii++)
			temp[iii] = randn(0, 1.0 / dimension, -6.0 / sqrt(dimension), 6.0 / sqrt(dimension));
		entv[i].push_back(temp);
	}

	fin = fopen((inPath + "triple2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &tripleTotal);
	trainHead = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainList = (Triple *)calloc(tripleTotal, sizeof(Triple));
	tripleTotal = 0;
	info_of_rel.resize(relationTotal);
	while (fscanf(fin, "%d", &trainList[tripleTotal].h) == 1) {
		tmp = fscanf(fin, "%d", &trainList[tripleTotal].t);
		tmp = fscanf(fin, "%d", &trainList[tripleTotal].r);
		trainHead[tripleTotal].h = trainList[tripleTotal].h;
		trainHead[tripleTotal].t = trainList[tripleTotal].t;
		trainHead[tripleTotal].r = trainList[tripleTotal].r;
		trainTail[tripleTotal].h = trainList[tripleTotal].h;
		trainTail[tripleTotal].t = trainList[tripleTotal].t;
		trainTail[tripleTotal].r = trainList[tripleTotal].r;
		Triple temp_triple;
		temp_triple.h = trainList[tripleTotal].h;
		temp_triple.r = trainList[tripleTotal].r;
		temp_triple.t = -1;
		info_of_rel[trainList[tripleTotal].r][temp_triple] ++;
		temp_triple.h = -1;
		temp_triple.r = trainList[tripleTotal].r;
		temp_triple.t = trainList[tripleTotal].t;
		info_of_rel[trainList[tripleTotal].r][temp_triple] ++;
		tripleTotal++;
	}
	fclose(fin);
	int countt1 = 0;
	int counth1 = 0;
	num_of_rel.resize(relationTotal);
	for(int i = 0; i < info_of_rel.size(); i++) {
		// printf("temp_triple %i: %i\n", i, (info_of_rel[i].size()));
		int maxh = 0, maxt = 0, counth = 0, countt = 0;
		double meanh = 0.0, meant = 0.0;
		for(map<Triple, int>::iterator it = info_of_rel[i].begin(); it != info_of_rel[i].end(); it++) {
			if(it->first.t == -1) {
				meant += (double) it->second;
				countt ++;
				if(it->second > maxt) maxt = it->second;
			}
			if(it->first.h == -1) {
				meanh += (double) it->second;
				counth ++;
				if(it->second > maxh) maxh = it->second;
			}
		}
		meant = meant/(double)countt;
		meanh = meanh/(double)counth;
		num_of_rel[i].maxh = maxh;
		num_of_rel[i].meanh = meanh;
		num_of_rel[i].maxt = maxt;
		num_of_rel[i].meant = meant;
		// printf("temp_triple[%i] maxtail: %i, meantail: %.3f, maxhead: %i, meanhead: %.3f\n", i, maxt, meant, maxh, meanh);
		if(maxh == 1) counth1++;
		if(maxt == 1) countt1++;
	}
	// printf("counth1: %i, countt1: %i\n", counth1, countt1);

	sort(trainHead, trainHead + tripleTotal, cmp_head());
	sort(trainTail, trainTail + tripleTotal, cmp_tail());

	lefHead = (int *)calloc(entityTotal, sizeof(int));
	rigHead = (int *)calloc(entityTotal, sizeof(int));
	lefTail = (int *)calloc(entityTotal, sizeof(int));
	rigTail = (int *)calloc(entityTotal, sizeof(int));
	memset(rigHead, -1, sizeof(rigHead));
	memset(rigTail, -1, sizeof(rigTail));
	for (int i = 1; i < tripleTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
	}
	rigHead[trainHead[tripleTotal - 1].h] = tripleTotal - 1;
	rigTail[trainTail[tripleTotal - 1].t] = tripleTotal - 1;

	// init mixture factor
	factor.resize(entityTotal);
	sigma.resize(entityTotal);
	count_for_factor.resize(entityTotal);
	for (size_t i = 0; i < entityTotal; i++) {
		factor[i].push_back(1.0);
		count_for_factor[i].push_back(1);
		sigma[i].push_back(10.0 / dimension);
	}
}

/*
	Training process of transE.
*/

int transeLen;
int transeBatch;
double res;

// (a+r-b).(a+r-b)
double norm(const vector<double> & a, const vector<double> & b, const vector<double> & r) {
	double sum = 0.0;
	double t;
	for (size_t i = 0; i < dimension; i++) {
		t = a[i]+r[i]-b[i];
		sum += t*t;
		// sum += pow(a[i]+r[i]-b[i], 2.0);
	}
	return sum;
}
// P(arb)
double prob(int a, int b, int r) {
	double p = 0.0;
	for (size_t ai = 0; ai < factor[a].size(); ai++) {
		for (size_t bi = 0; bi < factor[b].size(); bi++) {
			p += factor[a][ai]*factor[b][bi]*exp(-norm(entv[a][ai],entv[b][bi],relv[r])/(margin_e+sigma[a][ai]*sigma[a][ai]+sigma[b][bi]*sigma[b][bi]));
		}
	}
	if (isnan(entv[a][0][0])) {
		printf("%i,%i,%i\n", a,r,b);
		int ttt;
		scanf("prob entv %i\n", &ttt);
	}
	return p;
}
double prob_b(int a, int b, const vector<double> & r) {
	double p = 0.0;
	for (size_t ai = 0; ai < factor[a].size(); ai++) {
		for (size_t bi = 0; bi < factor[b].size(); bi++) {
			p += factor[a][ai]*factor[b][bi]*exp(-norm(entv[a][ai],entv[b][bi],r)/(2+sigma[a][ai]*sigma[a][ai]+sigma[b][bi]*sigma[b][bi]));
		}
	}
	return p;
}
int counter_loss = 0;
double loss_gradient(int a, int b, int r, bool sample_type) {
	double p = prob(a,b,r);
    if (p<0.01) {
        p = 0.01;
    }
    // cout << counter_loss++ << endl;
	double smn = 0.0;
	double x = 0.0;
	double local_const = (sample_type?1.0:-1.0)*transeAlpha*2.0/p;
	for (size_t m = 0; m < factor[a].size(); m++) {
		for (size_t n = 0; n < factor[b].size(); n++) {
			smn = factor[a][m]*factor[b][n]*exp(-norm(entv[a][m],entv[b][n],relv[r])/(margin_e+sigma[a][m]*sigma[a][m]+sigma[b][n]*sigma[b][n]))/(margin_e+sigma[a][m]*sigma[a][m]+sigma[b][n]*sigma[b][n]);
			x = local_const*norm(entv[a][m],entv[b][n],relv[r])*smn/(margin_e+sigma[a][m]*sigma[a][m]+sigma[b][n]*sigma[b][n]);
            if (isnan(x) || !isfinite(x)) {
                printf("sigma x is %lf\n", x);
                printf("%lf\n", p);
                printf("%lf %lf %lf %lf\n", local_const, norm(entv[a][m],entv[b][n],relv[r]), smn, (sigma[a][m]*sigma[a][m]+sigma[b][n]*sigma[b][n]));
                int tttt;
                scanf("%i\n", tttt);
            }
			// double p1, p2, delta;
			// p1 = -(sample_type?1.0:-1.0)*log(prob(a,b,r));
			// delta = x*sigma[a][m];
			sigma[a][m] += x*sigma[a][m];
			// p2 = -(sample_type?1.0:-1.0)*log(prob(a,b,r));
			// printf("sigma %lf %lf %lf %lf\n", (p2-p1)/delta/(-delta),x,prob(a,b,r),prob(a,b,r));


			sigma[b][n] += x*sigma[b][n];

			for (size_t j = 0; j < dimension; j++) {
				x = local_const*(entv[a][m][j]+relv[r][j]-entv[b][n][j])*smn;
                if (isnan(x)) {
                    printf("vector x is nan\n");
                    int tttt;
                    scanf("%i\n", tttt);
                }
				entv[a][m][j] -= x;

				// p1 = -(sample_type?1.0:-1.0)*log(prob(a,b,r));
				// delta = x;
				entv[b][n][j] += x;
				// p2 = -(sample_type?1.0:-1.0)*log(prob(a,b,r));
				// printf("entv b %lf %lf %lf %lf\n", (p2-p1)/delta/(-delta),x,p1,p2);
				relv[r][j] -= x;
                if (isnan(relv[r][j]) || !isfinite(relv[r][j])) {
                    printf("vector relv[r][j]is %lf %lf\n", relv[r][j], x);
                    // int tttt;
                    // scanf("%i\n", tttt);
                }
			}
		}
	}
	return (sample_type?-1.0:1.0)*log(p);
}
double regular_gradient(int a, int b, int r) {
	double sum = 0.0;
	for (size_t j = 0; j < dimension; j++) {
		sum += relv[r][j]*relv[r][j];
		relv[r][j] -= regular_c*transeAlpha*2.0*relv[r][j];
        if (isnan(relv[r][j]) || !isfinite(relv[r][j])) {
            printf("regular relv[r][j] is %lf\n", relv[r][j]);
        }
		for (size_t m = 0; m < factor[a].size(); m++) {
			for (size_t n = 0; n < factor[b].size(); n++) {
				sum += entv[a][m][j]*entv[a][m][j];
				sum += entv[b][n][j]*entv[b][n][j];
				entv[a][m][j] -= regular_c*transeAlpha*2.0*entv[a][m][j];
				entv[b][n][j] -= regular_c*transeAlpha*2.0*entv[b][n][j];
			}
		}
	}
	return regular_c*sum;
}
double generate(int a, int b, int r) {
	double p = prob(a,b,r);
	vector<double> r0(100,0.0);
	double p0 = prob_b(a,b,r0);
    // cout << p << " " << p0 << endl;
	// double crp_alpha = beta*p0/(beta*p0+p);
	double prob_of_gen = beta*p0/(beta*p0+p);
    // printf("%lf\n", prob_of_gen);
	// printf("p: %lf    prob_of_gen: %lf\n", p, prob_of_gen);
	double rr;
	int pos = 0;
	double sum;
	// b
    rr = rand(0.0,1.0);
    sum = 0.0;
    for (size_t i = 0; i < count_for_factor[b].size(); i++) {
        sum += count_for_factor[b][i];
    }
	// prob_of_gen = crp_alpha/(sum+crp_alpha);
    if (rr < prob_of_gen) {
        // generate
        new_vector_num++;
        // cout << a << " " << r << " " << b << " " << b << "new" << endl;
        count_for_factor[b].push_back(1);
		for (size_t i = 0; i < factor[b].size(); i++) {
            factor[b][i] = ((double)count_for_factor[b][i])/(sum+1.0);
        }
        factor[b].push_back(1.0/(sum+1.0));
        int max_factor_pos = 0;
        double max_factor_value = -1.0;
        for (size_t i = 0; i < factor[a].size(); i++) {
            if ( max_factor_value < factor[a][i] ) {
                max_factor_value = factor[a][i];
                max_factor_pos = i;
            }
        }
        vector<double> tempv;
        tempv.resize(dimension);
        for (size_t i = 0; i < dimension; i++) {
            tempv[i] = entv[a][max_factor_pos][i] + relv[r][i];
        }
        entv[b].push_back(tempv);
		sigma[b].push_back(sigma[a][max_factor_pos]);
    } else {
        // no generate
        rr = rand(0.0,sum);
        for (size_t i = 0; i < count_for_factor[b].size(); i++) {
            rr -= count_for_factor[b][i];
            if (rr<=0.0) {
                count_for_factor[b][i]++;
                break;
            }
        }
        for (size_t i = 0; i < factor[b].size(); i++) {
            factor[b][i] = ((double)count_for_factor[b][i])/(sum+1.0);
        }
    }
	// a
    rr = rand(0.0,1.0);
    sum = 0.0;
	for (size_t i = 0; i < count_for_factor[a].size(); i++) {
		sum += count_for_factor[a][i];
	}
	// prob_of_gen = crp_alpha/(sum+crp_alpha);
    if (rr < prob_of_gen) {
        // generate
        new_vector_num++;
        // cout << a << " " << r << " " << b << " " << a << "new" << endl;
        count_for_factor[a].push_back(1);
        for (size_t i = 0; i < factor[a].size(); i++) {
            factor[a][i] = ((double)count_for_factor[a][i])/(sum+1.0);
            // cout << "factor" << a << " " << i << " " << factor[a][i] << endl;
        }
        factor[a].push_back(1.0/(sum+1.0));
        int max_factor_pos = 0;
		double max_factor_value = -1.0;
		for (size_t i = 0; i < factor[b].size(); i++) {
			if ( max_factor_value < factor[b][i] ) {
				max_factor_value = factor[b][i];
				max_factor_pos = i;
			}
		}
		vector<double> tempv;
		tempv.resize(dimension);
		for (size_t i = 0; i < dimension; i++) {
			tempv[i] = entv[b][max_factor_pos][i] - relv[r][i];
		}
		entv[a].push_back(tempv);
		sigma[a].push_back(sigma[b][max_factor_pos]);

	} else {
        // no generate
        rr = rand(0.0,sum);
        for (size_t i = 0; i < count_for_factor[a].size(); i++) {
            rr -= count_for_factor[a][i];
    		if (rr<=0.0) {
    			count_for_factor[a][i]++;
    			break;
    		}
    	}
        for (size_t i = 0; i < factor[a].size(); i++) {
    		factor[a][i] = ((double)count_for_factor[a][i])/(sum+1.0);
        }

    }
}
void train_kb_mix(int a1, int b1, int r1, int a2, int b2, int r2) {
    double p1 = prob(a1,b1,r1);
    double p2 = prob(a2,b2,r2);
    if (isnan(p1) || isnan(p2)) {
        printf("%lf %lf\n", p1, p2);
        for (size_t i = 0; i < dimension; i++) {
            printf("%lf %lf %lf\n", entv[a1][0][i], relv[r1][i], entv[b1][0][i]);
        }
    }
	// if((p1/p2>margin_rate) && ((p1>0.5)) || (isnan(p1))) return;
    // if((p1/p2>margin_rate) && (p1>0.5)) return;
    if((p1/p2>margin_rate)) return;

    // printf("%lf %lf %lf %lf %lf %lf\n", p1, p2, entv[a1][0][0], entv[a2][0][0], sigma[a1][0], sigma[a2][0]);
    bad_triple ++;
	double x = 0.0;
	double smn = 0.0;
	double ttt = loss_gradient(a1, b1, r1, true);
	double fff = loss_gradient(a2, b2, r2, false);
	res += ttt;
	res += fff;
    // res += 1;
	regular_gradient(a1, b1, r1);
	// regular_gradient(a2, b2, r2);
    // generate(a1, b1, r1);
}


int corrupt_head(int id, int h, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefHead[h] - 1;
	rig = rigHead[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainHead[ll].t) return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

int corrupt_tail(int id, int t, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainTail[ll].h) return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

void* transetrainMode(void *con) {
	int id;
	id = (unsigned long long)(con);
	next_random[id] = rand();
	for (int k = transeBatch / transeThreads; k >= 0; k--) {
		int j;
		int i = rand_max(id, transeLen);
		int pr = 500;
		// if (randd(id) % 1000 < pr) {
			j = corrupt_head(id, trainList[i].h, trainList[i].r);
			train_kb_mix(trainList[i].h, trainList[i].t, trainList[i].r, trainList[i].h, j, trainList[i].r);
		// } else {
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
			train_kb_mix(trainList[i].h, trainList[i].t, trainList[i].r, j, trainList[i].t, trainList[i].r);
		// }
		// norm(relationVec + dimension * trainList[i].r);
		// norm(entityVec + dimension * trainList[i].h);
		// norm(entityVec + dimension * trainList[i].t);
		// norm(entityVec + dimension * j);
	}
}

void* train_transe(void *con) {
	transeLen = tripleTotal;
	transeBatch = transeLen / nbatches;
	next_random = (unsigned long long *)calloc(transeThreads, sizeof(unsigned long long));
	// for (int epoch = 0; epoch < 1; epoch++) {
	for (int epoch = 0; epoch < transeTrainTimes; epoch++) {
		res = 0;
        bad_triple = 0;
        new_vector_num = 0;
		printf("%i\n", epoch);
		for (int batch = 0; batch < nbatches; batch++) {
			pthread_t *pt = (pthread_t *)malloc(transeThreads * sizeof(pthread_t));
			for (int a = 0; a < transeThreads; a++)
				pthread_create(&pt[a], NULL, transetrainMode,  (void*)a);
			for (int a = 0; a < transeThreads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
		}
        if (epoch > 200) {
            for (size_t i = 0; i < tripleTotal; i++) {
                generate(trainList[i].h,trainList[i].t,trainList[i].r);
            }
        }

		// printf("epoch %d %lf\n", epoch, res);
        printf("epoch %d %lf bad_triple: %i new_vector_num: %i average_new: %lf average_res: %lf\n", epoch, res, bad_triple, new_vector_num, ((double)new_vector_num)/entityTotal, res/bad_triple);
		fflush(stdout);

	}
}


/*
	Get the results of transE.
*/

void out_transe() {
	FILE* f2 = fopen((outPath + "relation2vec."+version).c_str(), "w");
	FILE* f3 = fopen((outPath + "entity2vec."+version).c_str(), "w");
	FILE* f4 = fopen((outPath + "sigma."+version).c_str(), "w");
	for (int i=0; i < relationTotal; i++) {
		for (int ii = 0; ii < dimension; ii++)
			fprintf(f2, "%.6f\t", relv[i][ii]);
		fprintf(f2,"\n");
	}
	for (int i = 0; i < entityTotal; i++) {
		for (int j = 0; j < entv[i].size(); j++) {
			fprintf(f3, "%i\t", i);
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f3, "%.6f\t", entv[i][j][ii]);
			fprintf(f3,"\n");
		}

	}
	for (int i = 0; i < entityTotal; i++) {
		for (int j = 0; j < sigma[i].size(); j++) {
			fprintf(f3, "%i\t", i);
				fprintf(f3, "%.6f\t", sigma[i][j]);
			fprintf(f3,"\n");
		}

	}
	fclose(f2);
	fclose(f3);
	fclose(f4);
}

/*
	Main function
*/

int main() {
	init();
	train_transe(NULL);
	out_transe();
	return 0;
}
