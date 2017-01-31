#ifndef INIT_H
#define INIT_H
#include <cstring>
#include <cstdlib>
#include <vector>
#include <map>
#include <string>
#include <cstdio>
#include <float.h>
#include <cmath>

using namespace std;


string version = "";
int output_model = 0;

int num_threads = 10;
int trainTimes = 15;
float alpha = 0.02;
float reduce = 0.98;
int tt,tt1;

// Seems the size of convolutional layer.
int dimensionC = 230;//1000;

// Position feature dimension.
int dimensionWPE = 5;//25;

// Word window used to concatenate word embeddings.
int window = 3;

// Limit for distance between word and entity.
int limit = 30;


float marginPositive = 2.5;
float marginNegative = 0.5;
float margin = 2;

float Belt = 0.001;

float *matrixB1, *matrixRelation, *matrixW1, *matrixRelationDao, *matrixRelationPr, *matrixRelationPrDao;
float *matrixB1_egs, *matrixRelation_egs, *matrixW1_egs, *matrixRelationPr_egs;
float *matrixB1_exs, *matrixRelation_exs, *matrixW1_exs, *matrixRelationPr_exs;
float *wordVecDao,*wordVec_egs,*wordVec_exs;
float *positionVecE1, *positionVecE2, *matrixW1PositionE1, *matrixW1PositionE2;
float *positionVecE1_egs, *positionVecE2_egs, *matrixW1PositionE1_egs, *matrixW1PositionE2_egs, *positionVecE1_exs, *positionVecE2_exs, *matrixW1PositionE1_exs, *matrixW1PositionE2_exs;
float *matrixW1PositionE1Dao;
float *matrixW1PositionE2Dao;
float *positionVecDaoE1;
float *positionVecDaoE2;
float *matrixW1Dao;
float *matrixB1Dao;

double mx = 0;

int batch = 16;
int npoch;
int len;

float rate = 1;
FILE *logg;

float *wordVec;

// Word total number, word embedding dimension, relation type number.
int wordTotal, dimension, relationTotal;

// Token-to-entity distance min, max, range.
int PositionMinE1, PositionMaxE1, PositionTotalE1,PositionMinE2, PositionMaxE2, PositionTotalE2;

// Word to index mapping. (Index in embedding file).
map<string,int> wordMapping;

// All words in pre-trained word embeddings file.
vector<string> wordList;

// Relation name to id mapping.
map<string,int> relationMapping;

vector<int *> trainLists, trainPositionE1, trainPositionE2;
vector<int> trainLength;
vector<int> headList, tailList, relationList;
vector<int *> testtrainLists, testPositionE1, testPositionE2;
vector<int> testtrainLength;
vector<int> testheadList, testtailList, testrelationList;
vector<std::string> nam;

map<string,vector<int> > bags_train, bags_test;

void init() {
	FILE *f = fopen("../data/vec.bin", "rb");
	FILE *fout = fopen("../data/vector1.txt", "w");
	
	// Read word total number and word embedding dimension.
	fscanf(f, "%d", &wordTotal);
	fscanf(f, "%d", &dimension);
	cout<<"wordTotal=\t"<<wordTotal<<endl;
	cout<<"Word dimension=\t"<<dimension<<endl;
	
	
	PositionMinE1 = 0;
	PositionMaxE1 = 0;
	PositionMinE2 = 0;
	PositionMaxE2 = 0;
	
	// Allocate memory for pre-trained word embeddings.
	wordVec = (float *)malloc((wordTotal+1) * dimension * sizeof(float));
	
	// Allocate memory for all unique words.
	wordList.resize(wordTotal+1);
	wordList[0] = "UNK";
	
	// Loop through pre-trained word embeddings.
	for (int b = 1; b <= wordTotal; b++) {
		string name = "";
		
		// Read word.
		while (1) {
			char ch = fgetc(f);
			if (feof(f) || ch == ' ') break;
			if (ch != '\n') name = name + ch;
		}
		
		// Write word to file.
		fprintf(fout, "%s", name.c_str());
		
		// Find the index for current word in embedding array.
		int last = b * dimension;
		
		// Sum of square.
		float smp = 0;
		for (int a = 0; a < dimension; a++) {
			// Read a float number to embedding array.
			fread(&wordVec[a + last], sizeof(float), 1, f);
			// Add square of current float.
			smp += wordVec[a + last]*wordVec[a + last];
		}
		
		smp = sqrt(smp);
		
		// Normalize the embeddings by sum of square.
		for (int a = 0; a< dimension; a++)
		{
			wordVec[a+last] = wordVec[a+last] / smp;
			// Save the new embedding to file.
			fprintf(fout, "\t%f", wordVec[a+last]);
		}
		fprintf(fout,"\n");
		
		// Save word to index mapping.
		wordMapping[name] = b;
		// Save index to word mapping.
		wordList[b] = name;
	}
	fclose(fout);
	
	// Why add 1? - It adds 'UNK' for unknown words.
	wordTotal+=1;
	fclose(f);
	
	// Load relation to id mapping.
	char buffer[1000];
	f = fopen("../data/RE/relation2id.txt", "r");
	while (fscanf(f,"%s",buffer)==1) {
		int id;
		fscanf(f,"%d",&id);
		relationMapping[(string)(buffer)] = id;
		relationTotal++;
		nam.push_back((std::string)(buffer));
	}
	fclose(f);
	cout<<"relationTotal:\t"<<relationTotal<<endl;
	
	// Load training data.
	f = fopen("../data/RE/train.txt", "r");
	while (fscanf(f,"%s",buffer)==1)  {
		// First entity id.
		string e1 = buffer;
		
		fscanf(f,"%s",buffer);
		// Second entity id.
		string e2 = buffer;
		
		fscanf(f,"%s",buffer);
		// First entity string.
		string head_s = (string)(buffer);
		// Get id for first entity.
		int head = wordMapping[(string)(buffer)];
		
		fscanf(f,"%s",buffer);
		// Get second entity string and its id.
		int tail = wordMapping[(string)(buffer)];
		string tail_s = (string)(buffer);
		
		// This is relation type.
		fscanf(f,"%s",buffer);
		
		// It's actually building a map from <e1, e2, relation> to head/tail entity index.
		bags_train[e1+"\t"+e2+"\t"+(string)(buffer)].push_back(headList.size());
		
		// Find relation type id.
		int num = relationMapping[(string)(buffer)];
		
		// lefnum = token position in sentence for head.
		// rignum = token position in sentence for tail.
		int len = 0, lefnum = 0, rignum = 0;
		
		// A list of word ids for the sentence.
		std::vector<int> tmpp;
		
		// Read the sentence word by word.
		while (fscanf(f,"%s", buffer)==1) {
			std::string con = buffer;
			
			// End of sentence.
			if (con=="###END###") break;
			
			// Find word id.
			int gg = wordMapping[con];
			
			// Save the token position for head and tail.
			if (con == head_s) lefnum = len;
			if (con == tail_s) rignum = len;
			len++;
			
			// Store word id.
			tmpp.push_back(gg);
		}
		
		// Store head word id and tail word id. Why?
		headList.push_back(head);
		tailList.push_back(tail);
		
		// Store relation id.
		relationList.push_back(num);
		
		// Store sentence length.
		trainLength.push_back(len);
		
		// What these for?
		int *con=(int *)calloc(len,sizeof(int));
		int *conl=(int *)calloc(len,sizeof(int));
		int *conr=(int *)calloc(len,sizeof(int));
		
		// Loop sentence.
		for (int i = 0; i < len; i++) {
			// Word id in sentence.
			con[i] = tmpp[i];
			
			// Distance to head word.
			conl[i] = lefnum - i;
			
			// Distance to tail word.
			conr[i] = rignum - i;
			
			// Restrict the distance.
			if (conl[i] >= limit) conl[i] = limit;
			if (conr[i] >= limit) conr[i] = limit;
			if (conl[i] <= -limit) conl[i] = -limit;
			if (conr[i] <= -limit) conr[i] = -limit;
			
			// Record the max values.
			if (conl[i] > PositionMaxE1) PositionMaxE1 = conl[i];
			if (conr[i] > PositionMaxE2) PositionMaxE2 = conr[i];
			if (conl[i] < PositionMinE1) PositionMinE1 = conl[i];
			if (conr[i] < PositionMinE2) PositionMinE2 = conr[i];
		}
		// Each element is a list of the word id in the sentence.
		trainLists.push_back(con);
		// Each element is a list of distances between the words and head token.
		trainPositionE1.push_back(conl);
		// Each element is a list of distances between the words and tail token.
		trainPositionE2.push_back(conr);
	}
	fclose(f);

	// Read test data. Similar to above for training data.
	f = fopen("../data/RE/test.txt", "r");	
	while (fscanf(f,"%s",buffer)==1)  {
		// First entity id.
		string e1 = buffer;
		
		// Second entity id.
		fscanf(f,"%s",buffer);
		string e2 = buffer;
		
		// For what?
		bags_test[e1+"\t"+e2].push_back(testheadList.size());
		
		// First entity string.
		fscanf(f,"%s",buffer);
		string head_s = (string)(buffer);
		// First entity word id.
		int head = wordMapping[(string)(buffer)];
		
		// Second entity string
		fscanf(f,"%s",buffer);
		string tail_s = (string)(buffer);
		// Second entity word id.
		int tail = wordMapping[(string)(buffer)];
		
		// Relation name and id.
		fscanf(f,"%s",buffer);
		int num = relationMapping[(string)(buffer)];
		
		// Find token position for entities in sentence.
		int len = 0 , lefnum = 0, rignum = 0;
		// A list of word id in sentence.
		std::vector<int> tmpp;
		while (fscanf(f,"%s", buffer)==1) {
			std::string con = buffer;
			if (con=="###END###") break;
			int gg = wordMapping[con];
			if (head_s == con) lefnum = len;
			if (tail_s == con) rignum = len;
			len++;
			tmpp.push_back(gg);
		}
		
		// Store head and tail word id.
		testheadList.push_back(head);
		testtailList.push_back(tail);
		// Store relation id.
		testrelationList.push_back(num);
		// Store sentence length.
		testtrainLength.push_back(len);
		
		// Again compute the distances between words and the entities.
		int *con=(int *)calloc(len,sizeof(int));
		int *conl=(int *)calloc(len,sizeof(int));
		int *conr=(int *)calloc(len,sizeof(int));
		for (int i = 0; i < len; i++) {
			con[i] = tmpp[i];
			conl[i] = lefnum - i;
			conr[i] = rignum - i;
			if (conl[i] >= limit) conl[i] = limit;
			if (conr[i] >= limit) conr[i] = limit;
			if (conl[i] <= -limit) conl[i] = -limit;
			if (conr[i] <= -limit) conr[i] = -limit;
			if (conl[i] > PositionMaxE1) PositionMaxE1 = conl[i];
			if (conr[i] > PositionMaxE2) PositionMaxE2 = conr[i];
			if (conl[i] < PositionMinE1) PositionMinE1 = conl[i];
			if (conr[i] < PositionMinE2) PositionMinE2 = conr[i];
		}
		testtrainLists.push_back(con);
		testPositionE1.push_back(conl);
		testPositionE2.push_back(conr);
	}
	fclose(f);
	cout<<PositionMinE1<<' '<<PositionMaxE1<<' '<<PositionMinE2<<' '<<PositionMaxE2<<endl;

	// trainPosition E1 is distances between words and head entity.
	for (int i = 0; i < trainPositionE1.size(); i++) {
		// Sentence length.
		int len = trainLength[i];
		
		// Convert distance to non-negative values.
		int *work1 = trainPositionE1[i];
		for (int j = 0; j < len; j++)
			work1[j] = work1[j] - PositionMinE1;
		
		int *work2 = trainPositionE2[i];
		for (int j = 0; j < len; j++)
			work2[j] = work2[j] - PositionMinE2;
	}

	// Same as above.
	for (int i = 0; i < testPositionE1.size(); i++) {
		int len = testtrainLength[i];
		int *work1 = testPositionE1[i];
		
		// Convert distance to non-negative values.
		for (int j = 0; j < len; j++)
			work1[j] = work1[j] - PositionMinE1;
		int *work2 = testPositionE2[i];
		for (int j = 0; j < len; j++)
			work2[j] = work2[j] - PositionMinE2;
	}
	PositionTotalE1 = PositionMaxE1 - PositionMinE1 + 1;
	PositionTotalE2 = PositionMaxE2 - PositionMinE2 + 1;
}

// tanh activation function, similar to sigmoid.
float CalcTanh(float con) {
	if (con > 20) return 1.0;
	if (con < -20) return -1.0;
	float sinhx = exp(con) - exp(-con);
	float coshx = exp(con) + exp(-con);
	return sinhx / coshx;
}

// ?
float tanhDao(float con) {
	float res = CalcTanh(con);
	return 1 - res * res;
}

// sigmoid activation function.
float sigmod(float con) {
	if (con > 20) return 1.0;
	if (con < -20) return 0.0;
	con = exp(con);
	return con / (1 + con);
}

// ?
int getRand(int l,int r) {
	int len = r - l;
	int res = rand()*rand() % len;
	if (res < 0)
		res+=len;
	return res + l;
}

// Get random float between l and r.
float getRandU(float l, float r) {
	float len = r - l;
	// 0 <= res <= 1
	float res = (float)(rand()) / RAND_MAX;
	return res * len + l;
}

// Normalize ll to rr elements in a.
void norm(float* a, int ll, int rr)
{
	float tmp = 0;
	for (int i=ll; i<rr; i++)
		tmp+=a[i]*a[i];
	if (tmp>1)
	{
		tmp = sqrt(tmp);
		for (int i=ll; i<rr; i++)
			a[i]/=tmp;
	}
}


#endif
