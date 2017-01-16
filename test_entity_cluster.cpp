//////////////
#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<algorithm>
#include<cmath>
#include<cstdlib>

using namespace std;

bool debug=false;
bool L1_flag=1;

string version;
string trainortest = "test";
string dataPath = "../data/";
string vecPath = "./outvec/";
int topx = 10;
double margin_e = 0.1;


map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
map<string,string> mid2name,mid2type;
map<int,map<int,int> > entity2num;
map<int,int> e2num;
map<pair<string,string>,map<string,double> > rel_left,rel_right;

int relation_num,entity_num;

int dimension = 100, n = 100;

double sigmod(double x)
{
	return 1.0/(1+exp(-x));
}

double vec_len(vector<double> a)
{
	double res=0;
	for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	return sqrt(res);
}

void vec_output(vector<double> a)
{
	for (int i=0; i<a.size(); i++)
	{
		cout<<a[i]<<"\t";
		if (i%10==9)
			cout<<endl;
	}
	cout<<"-------------------------"<<endl;
}

double sqr(double x)
{
	return x*x;
}

char buf[100000],buf1[100000];

int my_cmp(pair<double,int> a,pair<double,int> b)
{
	return a.first>b.first;
}

double cmp(pair<int,double> a, pair<int,double> b)
{
	return a.second<b.second;
}
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
class Test{
	vector<vector<double> > relation_vec,entity_vec;
	vector<vector<double> > relv;
	multimap<int, vector<double> > headcluster, tailcluster;
	// multimap<int, vector<double> > entv;
	vector<vector<vector<double> > > entv;
	vector<vector<double> > factor; //mixture factor
	vector<vector<double> > sigma; // variance of an enity

	int cluster_num;


	vector<int> h,l,r;
	vector<int> fb_h,fb_l,fb_r;
	map<pair<int,int>, map<int,int> > ok;
	double res ;
public:
	void add(int x,int y,int z, bool flag)
	{
		if (flag)
		{
			fb_h.push_back(x);
			fb_r.push_back(z);
			fb_l.push_back(y);
		}
		ok[make_pair(x,z)][y]=1;
	}

	int rand_max(int x)
	{
		int res = (rand()*rand())%x;
		if (res<0)
			res+=x;
		return res;
	}
	double len;
// 	double calc_sum(int e1,int e2,int rel)
// 	{
// 		double sum=0;
// 		if (L1_flag)
// 			for (int ii=0; ii<n; ii++)
// 				sum+=-fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
// 		else
// 			for (int ii=0; ii<n; ii++)
// 				sum+=-sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
// 		return sum;
// 	}
// 	double calc_sum_multimap(int e1,int e2,int rel)
// 	{
// 		double sum=0.0,temp=100000000.0;
// 		pair<multimap<int,vector<double> >::iterator, multimap<int,vector<double> >::iterator>
// 				head = headcluster.equal_range(e1);
// 		pair<multimap<int,vector<double> >::iterator, multimap<int,vector<double> >::iterator>
// 				tail = tailcluster.equal_range(e2);
// 		multimap<int,vector<double> >::iterator it_head, it_tail;
// 		for(it_head = head.first; it_head != head.second; it_head++) {
// 			for(it_tail = tail.first; it_tail != tail.second; it_tail++) {
// 				sum = 0;
// 				if (L1_flag)
// 					for (int ii=0; ii<n; ii++)
// 						sum+=-fabs(it_tail->second[ii]-it_head->second[ii]-relation_vec[rel][ii]);
// 				else
// 					for (int ii=0; ii<n; ii++)
// 						sum+=-sqr(it_tail->second[ii]-it_head->second[ii]-relation_vec[rel][ii]);
// 				if(sum<temp) temp=sum;
// 			}
// 		}
// 		return temp;
// 	}
// 	vector<double> calc_sumv_multitail(int e1,int e2,int rel)
// 	{
// 		double sum=0.0,temp=100000000.0;
// 		int count = 0;
// 		vector<double> sumv;
// //		pair<multimap<int,vector<double> >::iterator, multimap<int,vector<double> >::iterator>
// //				head = headcluster.equal_range(e1);
// 		pair<multimap<int,vector<double> >::iterator, multimap<int,vector<double> >::iterator>
// 				tail = tailcluster.equal_range(e2);
// 		multimap<int,vector<double> >::iterator it_head, it_tail;
// //		for(it_head = head.first; it_head != head.second; it_head++) {
// 			for(it_tail = tail.first; it_tail != tail.second; it_tail++) {
// 				sum = 0;
// 				count ++;
// 				if (L1_flag)
// 					for (int ii=0; ii<n; ii++)
// 						sum+=-fabs(it_tail->second[ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
// 				else
// 					for (int ii=0; ii<n; ii++)
// 						sum+=-sqr(it_tail->second[ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
// //				if(sum<temp) temp=sum;
// 				sumv.push_back(sum);
// //				temp += sum;
// 			}
// //		}
// 		return sumv;
// 	}
// 	vector<double> calc_sumv_multihead(int e1,int e2,int rel)
// 	{
// 		double sum=0.0,temp=100000000.0;
// 		vector<double> sumv;
// 		int count = 0;
// 		pair<multimap<int,vector<double> >::iterator, multimap<int,vector<double> >::iterator>
// 				head = headcluster.equal_range(e1);
// //		pair<multimap<int,vector<double> >::iterator, multimap<int,vector<double> >::iterator>
// //				tail = tailcluster.equal_range(e2);
// 		multimap<int,vector<double> >::iterator it_head, it_tail;
// 		for(it_head = head.first; it_head != head.second; it_head++) {
// //			for(it_tail = tail.first; it_tail != tail.second; it_tail++) {
// 				sum = 0;
// 			count++;
// 				if (L1_flag)
// 					for (int ii=0; ii<n; ii++)
// 						sum+=-fabs(entity_vec[e2][ii]-it_head->second[ii]-relation_vec[rel][ii]);
// 				else
// 					for (int ii=0; ii<n; ii++)
// 						sum+=-sqr(entity_vec[e2][ii]-it_head->second[ii]-relation_vec[rel][ii]);
// //				if(sum<temp) temp=sum;
// 			sumv.push_back(sum);
// //			temp += sum;
// //			}
// 		}
// 		return sumv;
// 	}
// 	double calc_sum_multitail(int e1,int e2,int rel)
// 	{
// 		double sum=0.0,temp=100000000.0;
// 		int count = 0;
// //		pair<multimap<int,vector<double> >::iterator, multimap<int,vector<double> >::iterator>
// //				head = headcluster.equal_range(e1);
// 		pair<multimap<int,vector<double> >::iterator, multimap<int,vector<double> >::iterator>
// 				tail = tailcluster.equal_range(e2);
// 		multimap<int,vector<double> >::iterator it_head, it_tail;
// //		for(it_head = head.first; it_head != head.second; it_head++) {
// 		for(it_tail = tail.first; it_tail != tail.second; it_tail++) {
// 			sum = 0;
// 			count ++;
// 			if (L1_flag)
// 				for (int ii=0; ii<n; ii++)
// 					sum+=-fabs(it_tail->second[ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
// 			else
// 				for (int ii=0; ii<n; ii++)
// 					sum+=-sqr(it_tail->second[ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
// 				if(sum<temp) temp=sum;
// //				temp += sum;
// 		}
// //		}
// 		return temp;
// 	}
// 	double calc_sum_multihead(int e1,int e2,int rel)
// 	{
// 		double sum=0.0,temp=100000000.0;
// 		int count = 0;
// 		pair<multimap<int,vector<double> >::iterator, multimap<int,vector<double> >::iterator>
// 				head = headcluster.equal_range(e1);
// //		pair<multimap<int,vector<double> >::iterator, multimap<int,vector<double> >::iterator>
// //				tail = tailcluster.equal_range(e2);
// 		multimap<int,vector<double> >::iterator it_head, it_tail;
// 		for(it_head = head.first; it_head != head.second; it_head++) {
// //			for(it_tail = tail.first; it_tail != tail.second; it_tail++) {
// 			sum = 0;
// 			count++;
// 			if (L1_flag)
// 				for (int ii=0; ii<n; ii++)
// 					sum+=-fabs(entity_vec[e2][ii]-it_head->second[ii]-relation_vec[rel][ii]);
// 			else
// 				for (int ii=0; ii<n; ii++)
// 					sum+=-sqr(entity_vec[e2][ii]-it_head->second[ii]-relation_vec[rel][ii]);
// 				if(sum<temp) temp=sum;
// //			temp += sum;
// //			}
// 		}
// 		return temp;
// 	}

// prob
	// double calc_sum_prob(int a,int b,int r)
	double calc_sum(int a,int b,int r)
	{
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
	void run()
	{
		FILE* f1 = fopen((vecPath+"relation2vec."+version).c_str(),"r");
		// FILE* f3 = fopen((vecPath+"entity2vec."+version).c_str(),"r");
		cout<<relation_num<<' '<<entity_num<<endl;
		int relation_num_fb=relation_num;
		relation_vec.resize(relation_num_fb);
		for (int i=0; i<relation_num_fb;i++)
		{
			relation_vec[i].resize(n);
			for (int ii=0; ii<n; ii++)
				fscanf(f1,"%lf",&relation_vec[i][ii]);
		}
		relv = relation_vec;
		// entity_vec.resize(entity_num);
		// for (int i=0; i<entity_num;i++)
		// {
		// 	entity_vec[i].resize(n);
		// 	for (int ii=0; ii<n; ii++)
		// 		fscanf(f3,"%lf",&entity_vec[i][ii]);
		// 	//if (vec_len(entity_vec[i])-1>1e-3)
		// 	//	cout<<"wrong_entity"<<i<<' '<<vec_len(entity_vec[i])<<endl;
		// }
		fclose(f1);
		// fclose(f3);
		FILE* fin;
		vector<double> tempvec;
		tempvec.resize(n);
		int tempid;
		fin = fopen((vecPath+"entity2vec."+version).c_str(),"r");
		entv.resize(entity_num);
		while (fscanf(fin,"%i",&tempid)) {
			for (int ii=0; ii<n; ii++) fscanf(fin,"%lf",&tempvec[ii]);
			entv[tempid].push_back(tempvec);
		}
		fclose(fin);
		fin = fopen((vecPath+"sigma."+version).c_str(),"r");
		sigma.resize(entity_num);
		while (fscanf(fin,"%i",&tempid)) {
			double tempvalue;
			fscanf(fin,"%lf",&tempvalue);
			sigma[tempid].push_back(tempvalue);
		}
		fclose(fin);
		fin = fopen((vecPath+"factor."+version).c_str(),"r");
		factor.resize(entity_num);
		while (fscanf(fin,"%i",&tempid)) {
			double tempvalue;
			fscanf(fin,"%lf",&tempvalue);
			factor[tempid].push_back(tempvalue);
		}
		fclose(fin);
		// fin = fopen((vecPath+"headcluster.0_14950."+version).c_str(),"r");
		// fscanf(fin,"%i",&cluster_num);
		// for (int i=0; i<cluster_num;i++)
		// {
		// 	fscanf(fin,"%i",&tempid);
		// 	for (int ii=0; ii<n; ii++) fscanf(fin,"%lf",&tempvec[ii]);
		// 	headcluster.insert(make_pair(tempid,tempvec));
		// }
		// fclose(fin);
		// cout << headcluster.size() << endl;
		// fin = fopen((vecPath+"tailcluster.0_14950."+version).c_str(),"r");
		// fscanf(fin,"%i",&cluster_num);
		// for (int i=0; i<cluster_num;i++)
		// {
		// 	fscanf(fin,"%i",&tempid);
		// 	for (int ii=0; ii<n; ii++) fscanf(fin,"%lf",&tempvec[ii]);
		// 	tailcluster.insert(make_pair(tempid,tempvec));
		// }
		// fclose(fin);
		// cout << tailcluster.size() << endl;

		double lsum=0 ,lsum_filter= 0;
		double rsum = 0,rsum_filter=0;
		double lp_n=0,lp_n_filter;
		double rp_n=0,rp_n_filter;
		map<int,double> lsum_r,lsum_filter_r;
		map<int,double> rsum_r,rsum_filter_r;
		map<int,double> lp_n_r,lp_n_filter_r;
		map<int,double> rp_n_r,rp_n_filter_r;
		map<int,int> rel_num;

		for (int testid = 0; testid<fb_l.size(); testid+=1)
		{
			int h = fb_h[testid];
			int l = fb_l[testid];
			int rel = fb_r[testid];
			double tmp = calc_sum(h,l,rel);
			// if (testid == 0) {
			//     printf("%i %i %i\n", h, l, rel);
			// }
			// if (testid > 0) {
			//     break;
			// }
			rel_num[rel]+=1;
			int is_right = 0;
			vector<pair<int,double> > a;
			double ttt=0;
			int filter = 0;
			for (int i=0; i<entity_num; i++)
			{
				double sum = calc_sum(i,l,rel);
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			for (int i=a.size()-1; i>=0; i--)
			{
				if (ok[make_pair(a[i].first,rel)].count(l)>0)
					ttt++;
				if (ok[make_pair(a[i].first,rel)].count(l)==0)
					filter+=1;
				if (a[i].first ==h)
				{
					lsum+=a.size()-i;
					lsum_filter+=filter+1;
					lsum_r[rel]+=a.size()-i;
					lsum_filter_r[rel]+=filter+1;
					if (a.size()-i<=topx)
					{
						lp_n+=1;
						lp_n_r[rel]+=1;
					}
					if (filter<topx)
					{
						lp_n_filter+=1;
						lp_n_filter_r[rel]+=1;
					}
					break;
				}
			}
			a.clear();
			double mt = 0.0;
			for (int i=0; i<entity_num; i++)
				//for (int i=0; i<100; i++)
			{
				double sum = calc_sum(h,i,rel);
				// mt += sum/entity_num;
				// cout << h << " " << i << " " << rel << " " << sum << endl;
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			//cout << a[0].first << " " << a[0].second << endl;
			//cout << a[entity_num-1].first << " " << a[entity_num-1].second << endl;
			// for (size_t ii = 0; ii < a.size(); ii++) {
			//     if (a[ii].first == 14950) cout << ii << " " << a[ii].first << " " << a[ii].second << endl;
			// }
			ttt=0;
			filter=0;
			for (int i=a.size()-1; i>=0; i--)
			{
				if (ok[make_pair(h,rel)].count(a[i].first)>0)
					ttt++;
				if (ok[make_pair(h,rel)].count(a[i].first)==0)
					filter+=1;
				if (a[i].first==l)
				{
					rsum+=a.size()-i;
					rsum_filter+=filter+1;
					rsum_r[rel]+=a.size()-i;
					rsum_filter_r[rel]+=filter+1;
					if (a.size()-i<=topx)
					{
						rp_n+=1;
						rp_n_r[rel]+=1;
					}
					if (filter<topx)
					{
						rp_n_filter+=1;
						is_right = 1;
						rp_n_filter_r[rel]+=1;
					}
					break;
				}
			}
//			if (testid%100==0) cout<<testid<<":"<<"\t"<<lsum/(testid+1)<<' '<<lp_n/(testid+1)<<' '<<rsum/(testid+1)<<' '<<rp_n/(testid+1)<<"\t"<<lsum_filter/(testid+1)<<' '<<lp_n_filter/(testid+1)<<' '<<rsum_filter/(testid+1)<<' '<<rp_n_filter/(testid+1)<<endl;
			// if(is_right == 1) {
			// 	if (testid%100==0) cout<<testid<<":"<<"\t"<<lsum/(testid+1)<<' '<<lp_n/(testid+1)<<' '<<rsum/(testid+1)<<' '<<rp_n/(testid+1)<<"\t"<<lsum_filter/(testid+1)<<' '<<lp_n_filter/(testid+1)<<' '<<rsum_filter/(testid+1)<<' '<<rp_n_filter/(testid+1)<<endl;
			// 	is_right = 0;
			// 	continue;
			// }
			// a.clear();
			// for (int i=0; i<entity_num; i++)
			// {
			// 	double sum = calc_sum_prob(i,l,rel);
			// 	a.push_back(make_pair(i,sum));
			// 	// vector<double> sumv = calc_sumv_multitail(i,l,rel);
			// 	// for(int ii = 0; ii < sumv.size(); ii++) {
			// 	// 	a.push_back(make_pair(i,sumv[ii]));
			// 	// }
			// }
			// sort(a.begin(),a.end(),cmp);
			// ttt=0;
			// filter = 0;
			// for (int i=a.size()-1; i>=0; i--)
			// {
			// 	if (ok[make_pair(a[i].first,rel)].count(l)>0)
			// 		ttt++;
			// 	if (ok[make_pair(a[i].first,rel)].count(l)==0)
			// 		filter+=1;
			// 	if (a[i].first ==h)
			// 	{
			// 		lsum+=a.size()-i;
			// 		lsum_filter+=filter+1;
			// 		lsum_r[rel]+=a.size()-i;
			// 		lsum_filter_r[rel]+=filter+1;
			// 		if (a.size()-i<=topx)
			// 		{
			// 			lp_n+=1;
			// 			lp_n_r[rel]+=1;
			// 		}
			// 		if (filter<topx)
			// 		{
			// 			lp_n_filter+=1;
			// 			lp_n_filter_r[rel]+=1;
			// 		}
			// 		break;
			// 	}
			// }
			// a.clear();
			// for (int i=0; i<entity_num; i++)
			// {
			// 	double sum = calc_sum_prob(h,i,rel);
			// 	a.push_back(make_pair(i,sum));
			// 	// vector<double> sumv = calc_sumv_multihead(h,i,rel);
			// 	// for(int ii = 0; ii < sumv.size(); ii++) {
			// 	// 	a.push_back(make_pair(i,sumv[ii]));
			// 	// }
			// }
			// sort(a.begin(),a.end(),cmp);
			// ttt=0;
			// filter=0;
			// for (int i=a.size()-1; i>=0; i--)
			// {
			// 	if (ok[make_pair(h,rel)].count(a[i].first)>0)
			// 		ttt++;
			// 	if (ok[make_pair(h,rel)].count(a[i].first)==0)
			// 		filter+=1;
			// 	if (a[i].first==l)
			// 	{
			// 		rsum+=a.size()-i;
			// 		rsum_filter+=filter+1;
			// 		rsum_r[rel]+=a.size()-i;
			// 		rsum_filter_r[rel]+=filter+1;
			// 		if (a.size()-i<=topx)
			// 		{
			// 			rp_n+=1;
			// 			rp_n_r[rel]+=1;
			// 		}
			// 		if (filter<topx)
			// 		{
			// 			rp_n_filter+=1;
			//
			// 			rp_n_filter_r[rel]+=1;
			// 		}
			// 		break;
			// 	}
			// }
			if (testid%100==0) cout<<testid<<":"<<"\t"<<lsum/(testid+1)<<' '<<lp_n/(testid+1)<<' '<<rsum/(testid+1)<<' '<<rp_n/(testid+1)<<"\t"<<lsum_filter/(testid+1)<<' '<<lp_n_filter/(testid+1)<<' '<<rsum_filter/(testid+1)<<' '<<rp_n_filter/(testid+1)<<endl;




		}
		cout<<"left:"<<lsum/fb_l.size()<<'\t'<<lp_n/fb_l.size()<<"\t"<<lsum_filter/fb_l.size()<<'\t'<<lp_n_filter/fb_l.size()<<endl;
		cout<<"right:"<<rsum/fb_r.size()<<'\t'<<rp_n/fb_r.size()<<'\t'<<rsum_filter/fb_r.size()<<'\t'<<rp_n_filter/fb_r.size()<<endl;



	}

};
Test test;

void prepare()
{
	FILE* f1 = fopen((dataPath+"entity2id_o.txt").c_str(),"r");
	FILE* f2 = fopen((dataPath+"relation2id_o.txt").c_str(),"r");
	int x;
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{

		string st=buf;
		// printf("%s %i \n", st.c_str(), x);
		entity2id[st]=x;
		id2entity[x]=st;
		mid2type[st]="None";
		entity_num++;
	}
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		id2relation[x]=st;
		relation_num++;
	}
	FILE* f_kb = fopen((dataPath+"test.txt").c_str(),"r");
	while (fscanf(f_kb,"%s",buf)==1)
	{
		string s1=buf;
		fscanf(f_kb,"%s",buf);
		string s2=buf;
		fscanf(f_kb,"%s",buf);
		string s3=buf;
		if (entity2id.count(s1)==0)
		{
			cout<<"miss entity:"<<s1<<endl;
		}
		if (entity2id.count(s2)==0)
		{
			cout<<"miss entity:"<<s2<<endl;
		}
		if (relation2id.count(s3)==0)
		{
			cout<<"miss relation:"<<s3<<endl;
			relation2id[s3] = relation_num;
			relation_num++;
		}
		test.add(entity2id[s1],entity2id[s2],relation2id[s3],true);
	}
	fclose(f_kb);
	FILE* f_kb1 = fopen((dataPath+"train.txt").c_str(),"r");
	while (fscanf(f_kb1,"%s",buf)==1)
	{
		string s1=buf;
		fscanf(f_kb1,"%s",buf);
		string s2=buf;
		fscanf(f_kb1,"%s",buf);
		string s3=buf;
		if (entity2id.count(s1)==0)
		{
			cout<<"miss entity:"<<s1<<endl;
		}
		if (entity2id.count(s2)==0)
		{
			cout<<"miss entity:"<<s2<<endl;
		}
		if (relation2id.count(s3)==0)
		{
			relation2id[s3] = relation_num;
			relation_num++;
		}

		entity2num[relation2id[s3]][entity2id[s1]]+=1;
		entity2num[relation2id[s3]][entity2id[s2]]+=1;
		e2num[entity2id[s1]]+=1;
		e2num[entity2id[s2]]+=1;
		test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
	}
	fclose(f_kb1);
	FILE* f_kb2 = fopen((dataPath+"valid.txt").c_str(),"r");
	while (fscanf(f_kb2,"%s",buf)==1)
	{
		string s1=buf;
		fscanf(f_kb2,"%s",buf);
		string s2=buf;
		fscanf(f_kb2,"%s",buf);
		string s3=buf;
		if (entity2id.count(s1)==0)
		{
			cout<<"miss entity:"<<s1<<endl;
		}
		if (entity2id.count(s2)==0)
		{
			cout<<"miss entity:"<<s2<<endl;
		}
		if (relation2id.count(s3)==0)
		{
			relation2id[s3] = relation_num;
			relation_num++;
		}
		test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
	}
	fclose(f_kb2);
}


int main(int argc,char**argv)
{
	if (argc<2)
		return 0;
	else
	{
		version = argv[1];
		prepare();
		test.run();
	}
}
