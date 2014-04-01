#include <stdio.h>
#include <math.h>

#define MAX_RATINGS       100001
#define MAX_MOVIES        1683
#define MAX_CUSTOMERS     944
#define MIN_EPOCH         120
#define MAX_EPOCH         200
#define MAX_FEATURES      50
#define MIN_IMPROVEMENT   0.0001        // Minimum improvement required to continue current feature
#define INIT              0.1           // Initialization value for features
#define LRATE             0.001         // Learning rate parameter
#define K                 0.015         // Regularization parameter used to minimize over-fitting
#define PseudoCount       25.0

typedef unsigned char BYTE;
const char* TRAINING_FILE  =    "D:\\AI\\ml-100k\\u1.base";
const char* TESTING_FILE   =    "D:\\AI\\ml-100k\\u1.test";

struct Movies{
    int         RatingCount;
    int         RatingSum;
    double      RatingAvg;            
    double      PseudoAvg;            // Weighted average used to deal with small movie counts 
};

struct Customers{
    int         RatingCount;
    int         RatingSum;
};

struct Ratings{
    int         CustId;
    short       MovieId;
    BYTE        Rating;
    float       Cache;
};

class Recommender{
private:
  int N_rating_count;
  Ratings   m_Rating[MAX_RATINGS];
  Movies    m_Movie[MAX_MOVIES];
  Customers m_Customer[MAX_CUSTOMERS];
  double    Movie_Features[MAX_MOVIES][MAX_FEATURES];
  double    Customer_Features[MAX_CUSTOMERS][MAX_FEATURES];
public:
  Recommender();
  ~Recommender() { };
  void test();
  void Process_Data();
  void train();
  double Predict_Rating(int,int,int,double,bool);
  double Predict_Rating(int,int);
};

int main()
{
  Recommender* recommend = new Recommender();
  recommend->Process_Data();
  recommend->train();
  recommend->test();
return 0;
}

Recommender::Recommender()
{
  int f,i;
  N_rating_count = 0;
  for(f=0;f<MAX_FEATURES;f++)
  {
    for(i=0;i<MAX_MOVIES;i++)Movie_Features[i][f]=(float)INIT;
    for(i=0;i<MAX_CUSTOMERS;i++)Customer_Features[i][f]=(float)INIT;
  }
}

void Recommender::train()
{
  int f,e,i;
  double rmse,rmse_last,sq,err,p;
  double cf,mf;
  rmse = 4;
  FILE *fp;
  fp = fopen("epoch_rmse.csv","w");
  if(fp==NULL)
    printf("Can't open file for writing epoch error");
  for(f=0;f<MAX_FEATURES;f++)
  {
    for(e=0;(e<MIN_EPOCH)||(rmse<=rmse_last-MIN_IMPROVEMENT);e++)
    {
      sq = 0;
      rmse_last = rmse;
      for(i=0;i<N_rating_count;i++)
      {
        p = Predict_Rating(m_Rating[i].MovieId,m_Rating[i].CustId,f,m_Rating[i].Cache,true);
        err = (1.0*m_Rating[i].Rating-p);
        sq += err*err;
        cf = Customer_Features[m_Rating[i].CustId][f];
        mf = Movie_Features[m_Rating[i].MovieId][f];
        Customer_Features[m_Rating[i].CustId][f] += (double)(LRATE*(err*mf-K*cf));
        Movie_Features[m_Rating[i].MovieId][f]   += (double)(LRATE*(err*cf-K*mf));
      }
      rmse = sqrt(sq/N_rating_count);
      fprintf(fp,"%d,%d,%f\n",f,e,rmse);
    }
    
    for(i=0;i<N_rating_count;i++)
      m_Rating[i].Cache = Predict_Rating(m_Rating[i].MovieId,m_Rating[i].CustId,f,m_Rating[i].Cache,false);
    
  }
  fclose(fp);
}

void Recommender::test()
{
  int testCount = 0;
  FILE *fp;
  fp = fopen(TESTING_FILE,"r");
  if(fp==NULL)
    printf("Can't open training file");
  int user,movie,rating,time_stamp,i;
  double p,err,sq,rmse;
  sq = 0;
  while(fscanf(fp,"%d%d%d%d",&user,&movie,&rating,&time_stamp)!=EOF)
  {
      testCount++;
    p = Predict_Rating(movie,user);
    err = (1.0*rating-p);
    sq += err*err;
  }
  rmse = sqrt(sq/testCount);
  printf("Root mean square error in test set=%f",rmse);
  fclose(fp);
}

double Recommender::Predict_Rating(int movie,int user,int feature,double cache,bool bTrail)
{
  double sum = cache>0?cache:1;
  sum += Movie_Features[movie][feature]*Customer_Features[user][feature];
  if(sum>5) sum = 5;
  if(sum<1) sum = 1;
  
  if(bTrail)
  {
    sum += ((MAX_FEATURES-feature-1)*(INIT*INIT));
    if(sum>5) sum = 5;
    if(sum<1) sum = 1;
  }
  
  return sum;
}

double Recommender::Predict_Rating(int movie,int user)
{
  int i;
  double sum = 1;
  for(i=0;i<MAX_FEATURES;i++)
  {
    sum += Movie_Features[movie][i]*Customer_Features[user][i];
    if(sum>5) sum = 5;
    if(sum<1) sum = 1;
  }
  return sum;
}

void Recommender::Process_Data()
{
  FILE *fp;
  int user,movie,rating,time_stamp,i;
  
  for(i=0;i<MAX_MOVIES;i++)
  {
    m_Movie[i].RatingCount = 0;
    m_Movie[i].RatingSum   = 0;
  }
  
  for(i=0;i<MAX_CUSTOMERS;i++)
  {
    m_Customer[i].RatingCount = 0;
    m_Customer[i].RatingSum   = 0;
  }
  
  unsigned long long total_rating = 0;
  fp = fopen(TRAINING_FILE,"r");
  if(fp==NULL)
    printf("Can't open training file");
  
  while(fscanf(fp,"%d%d%d%d",&user,&movie,&rating,&time_stamp)!=EOF)
  {
    m_Rating[N_rating_count].CustId = user;
    m_Rating[N_rating_count].MovieId = movie;
    m_Rating[N_rating_count].Rating = rating;
    m_Rating[N_rating_count].Cache = 0.0;
    m_Movie[movie].RatingCount++;
    m_Movie[movie].RatingSum+=rating;
    m_Customer[user].RatingCount++;
    m_Customer[user].RatingSum+=rating;
    total_rating+=rating;
    N_rating_count++;
  }
  fclose(fp);
  float total_rating_avg = (1.0*total_rating)/(1.0*N_rating_count);
  for(i=0;i<MAX_MOVIES;i++)
  {
    m_Movie[i].RatingAvg = (1.0*m_Movie[i].RatingSum)/(1.0*m_Movie[i].RatingCount);
    m_Movie[i].PseudoAvg = (total_rating_avg*PseudoCount + 1.0*m_Movie[i].RatingSum)/(PseudoCount + 1.0*m_Movie[i].RatingCount);
  }
}
