//V5
#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fstream>
#include <sstream>
using namespace std;
class cell{public:
    double data,delt,dataUnacti;
    vector<cell*> prev;
    vector<double> weight,mt,vt;
cell(vector<cell*> a,
    vector<double> w,
    vector<double> m):
    
    data(0),delt(0),
    dataUnacti(0),prev(a),
    weight(w),mt(m),vt(m) {}};
#define SIZE 50000
#define max 3 //3
#define initialize 2.31 //2.31
#define relationship 3*in[i][0]+7*in[i][1]
#define file_relationship 3*x+7*y
#define netsize 2,3,1
#define LearningRate  rateRec=rateO*((sqrt(1-Bv))/(1-Bm));
#define threads 4
#define threadRate 0.0005
string equation = "3x + 7y";
class network{public:bool flag;
double rateO,rateRec,bias,Bm,Bv;
    vector<vector<cell*> > neuron;
    network (vector<int> topo,bool mul) {
       rateRec=rateO=0.00146;//rO:0.00146
        Bm=0.9;Bv=0.999;bias=0;flag=mul;
    for (int i=0;i<topo.size();i++)
    {vector <cell*> c;
    for(int j=0;j<topo[i];j++)
      {vector <double> v,mt(topo[i],0);
for(int k=0;k<topo[i];k++)
 v.push_back(initialize*(rand()/(double)RAND_MAX));
      if(i) c.push_back(new cell (neuron[i-1],v,mt));
   else c.push_back(new cell(c,v,mt));}
neuron.push_back(c);}}
    //end of class constructor
    void propaForward(vector<double> in){  
       for(int j=0;j<neuron[0].size();j++)
        neuron[0][j]->data=in[j];
        for (int i=1;i<neuron.size();i++)
        for(int j=0;j<neuron[i].size();j++)
         {cell* current=neuron[i][j];
                current->dataUnacti=0;
for(int temp=0;
           temp<current->prev.size();
                temp++)
       current->dataUnacti+= current->weight[temp]*current->prev[temp]->data;
 current->data=(i<neuron.size()-1)? acti(current->dataUnacti+bias):
actiOut(current->dataUnacti);}}
   
double acti(double x){//swish by tanhx
return (x>=0)?x:x*(0.5*tanh(0.5*x)+0.5);}
    
   double actiDe(double x){
return(x>0)?1:(0.5*tanh(0.5*x)+0.5)*
(x+1-x*(0.5*tanh(0.5*x)+0.5));}
   /*
    double acti(double x){//ReLU
      return(x>=0)?x:0;}
    
    double actiDe(double x){
      return(x>=0)?1:0;}
 
    double acti(double x){//swish by fast sig
      return
(x>=0)?x:x*(0.25*x/sqrt(1+0.25*x*x)+0.5);}
    
    double actiDe(double x){
      return(x>0)?1:
   (0.25*x/sqrt(1+0.25*x*x)+0.5)*(1+x*(1-(0.25*x/sqrt(1+0.25*x*x)+0.5)));}
    */
    double actiOut(double x){return x;}
    
    void calcError(vector <double> outEx){
        for(int i=0;i<neuron.size();i++)
        for(int j=0;j<neuron[i].size();j++)
        neuron[i][j]->delt=(i<neuron.size()-1)?
        0:(flag)?outEx[j]:(outEx[j]-neuron.back()[j]->data);
        for(int i=neuron.size()-1;i;i--)
        for(int j=0;j<neuron[i].size();j++)
for(int temp=0;
    temp<neuron[i][j]->prev.size();temp++)
neuron[i][j]->prev[temp]->delt-=
neuron[i][j]->weight[temp]*neuron[i][j]->delt;}
    
    void weightUpdate(){
        for(int i=1;i<neuron.size();i++)
       for(int j=0;j<neuron[i].size();j++)
         for(int temp=0;
           temp<neuron[i][j]->weight.size()&&
           temp<neuron[i][j]->prev.size();
              temp++)
    {//cout<<neuron[i][j]->weight[temp]<<endl;
       double st=(i<neuron.size()-1)?
 actiDe(neuron[i][j]->dataUnacti):1,
dLdw=neuron[i][j]->prev[temp]->data*st;
dLdw=0.5*neuron[i][j]->delt/dLdw;
    neuron[i][j]->mt[temp]=
      0.9*neuron[i][j]->mt[temp]+0.1*dLdw;     
    neuron[i][j]->vt[temp]=
  0.999*neuron[i][j]->vt[temp]+0.001*dLdw*dLdw;            
   neuron[i][j]->weight[temp]+=rateRec
     *(neuron[i][j]->mt[temp]/(1-Bm))/(1e-8+sqrt( neuron[i][j]->vt[temp]/(1-Bv)));}}
    
    void propaBack(vector <double> outEx){
        calcError(outEx);weightUpdate();}
    
    void train(vector<vector<double> > in,vector<vector<double> > out)
      {for (int i=0;i<in.size();i++)
        {propaForward(in[i]);
         propaBack(out[i]);
         Bv*=0.999;Bm*=0.9; 
          LearningRate
         cout<<"trained: "<<i+1<<"   error: "<<neuron.back()[0]->delt<<endl;}}
    
    void getResult()
        {vector <double> re;double u;
        cout<<"\ninput x, y\n";
     for (;re.size()<neuron[0].size();re.push_back(u))
          cin>>u;
         propaForward(re);
         for(int i=0;
        i<neuron[neuron.size()-1].size();i++)
  cout<<neuron.back()[i]->data<<" ";}
};
class Multi{public: 
    vector <network*> thread;
    vector <vector<double> > weight,mt,vt;
    vector <double> output;
    double rateO,rateRec,bias,Bm,Bv;int insize;
    Multi (vector<int> topo,int channel) {
       rateRec=rateO=threadRate;
        Bm=0.9;Bv=0.999;insize=topo[0];
      for(int i=0;i<channel;i++)
{vector<double> wl,ml;
 thread.push_back(new network(topo,1));
 for(int i=0;i<topo[topo.size()-1];i++)
 {wl.push_back(initialize*(rand()/(double)RAND_MAX));
  ml.push_back(0);}
 weight.push_back(wl);
 mt.push_back(ml);vt.push_back(ml);}
      for(int i=0;i<topo[topo.size()-1];i++)
      output.push_back(0);}
    
void forward(vector<double> in){
      for(int i=0;i<thread.size();i++)
      thread[i]->propaForward(in);
      for (int i=0;i<output.size();i++)
      output[i]=0;
      for (int i=0;i<output.size();i++) 
      for (int k=0;k<weight.size();k++)      
      output[i]+=weight[k][i]*thread[k]->neuron[
       thread[k]->neuron.size()-1][i]->data;}
void backward(vector <double> outEx){
    vector<vector<double> > error; 
    for (int k=0;k<weight.size();k++) 
    {vector<double> el;
    for (int i=0;i<output.size();i++)  el.push_back(weight[k][i]*(outEx[i]-output[i]));
    error.push_back(el);
    thread[k]->propaBack(el);}
    for (int k=0;k<weight.size();k++) 
    for (int i=0;i<output.size();i++)    
    {double dLdw=0.5*error[k][i]/(thread[k]->neuron.back()[i]->data*threads);
    mt[k][i]=0.9*mt[k][i]+0.1*dLdw;     
    vt[k][i]=vt[k][i]+0.001*dLdw*dLdw;            
    weight[k][i]+=rateRec
     *(mt[k][i]/(1-Bm))/(1e-8+sqrt(vt[k][i]/(1-Bv)));}}
void train(vector<vector<double> > in,vector<vector<double> > out)
        {for (int i=0;i<in.size();i++)
        {forward(in[i]);
         backward(out[i]);
         Bv*=0.999;Bm*=0.9; 
          LearningRate
         cout<<"trained: "<<i+1<<"   error: "<<out[i][0]-output[0]<<endl;}}
void getResult()
        {vector <double> re;double u;
        cout<<"\ninput x, y\n";
     for (;re.size()<insize;re.push_back(u))
          cin>>u;
         forward(re);
         for(int i=0;
        i<output.size();i++)
  cout<<output[i]<<" ";}
};
void genData(vector<vector<double> >& in,vector<vector<double> >& out)
{
   for (int i=0;i<SIZE;i++)
   {vector <double> k,v;
    for (int j=0;j<2;j++) 
     k.push_back(max*(rand()/(double)RAND_MAX));
     in.push_back(k);
     v.push_back(relationship);
      out.push_back(v);}}
int main(){srand(time(NULL));
vector<vector<double> > in, out;
vector<int> v;
for(int i=0,A[]={netsize};
    i-sizeof(A)/sizeof(int);
    v.push_back(A[i++]));
     Multi n0(v,threads);
     network n1(v,0);
 genData(in,out);
n0.train(in,out);
cout<<in[0][0]<<","<<in[0][1]
<<endl<<out[0][0]<<
"\ntraining relationship: output = "<<equation;
for(int r=0;r<5;r++)n0.getResult();
    
n1.train(in,out);
cout<<in[0][0]<<","<<in[0][1]
<<endl<<out[0][0]<<
"\ntraining relationship: output = "<<equation;
for(int r=0;r<5;r++)n1.getResult();       
    return 0;}