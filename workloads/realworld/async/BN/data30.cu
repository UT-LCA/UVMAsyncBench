// The data are synthesized.

#ifndef _DATA_H_
#define _DATA_H_
const int NODE_N=30;
const int STATE_N=2;
const int DATA_N=600;
int data[DATA_N*NODE_N]={
1,1,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,1,1,0,1,1,0,
1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,0,1,1,0,0,1,0,0,0,1,1,0,0,1,1,
1,1,0,1,0,1,0,0,1,0,0,0,0,1,1,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,
1,1,1,0,1,0,1,0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,0,0,0,1,0,0,0,1,
1,0,0,1,1,0,0,1,0,0,0,1,0,1,1,1,1,0,1,0,0,0,0,1,1,1,0,1,0,1,
1,0,0,0,0,0,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1,0,0,1,0,0,1,0,0,0,
1,1,0,1,1,0,1,0,0,0,0,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,
1,0,0,1,0,0,0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,
1,0,1,1,1,0,0,0,1,1,1,0,1,1,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,1,
1,1,1,1,1,0,1,1,0,1,0,1,1,1,0,0,0,0,0,1,1,1,1,0,1,0,0,0,1,1,
1,0,0,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,0,1,0,0,1,0,0,0,0,0,0,1,
1,0,0,1,0,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,
1,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,1,
1,0,0,1,0,0,1,0,0,1,0,1,0,1,1,1,0,1,1,1,0,0,1,0,0,0,1,0,1,1,
1,1,0,1,0,0,1,0,0,0,0,1,1,1,0,1,0,0,1,1,0,1,1,0,1,1,1,0,0,0,
1,0,0,1,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,1,1,0,0,1,1,0,0,
1,0,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,0,
1,1,1,1,0,1,1,0,0,0,0,1,1,0,0,1,1,1,0,1,0,0,1,0,1,1,1,0,1,0,
1,1,1,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,0,1,1,0,1,1,0,0,
1,1,1,1,0,0,1,0,1,0,1,1,1,0,0,1,0,1,0,0,1,1,1,0,1,0,1,1,0,0,
0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,0,1,1,0,0,1,1,1,0,1,0,0,0,0,1,
1,1,0,0,0,0,1,1,0,1,1,1,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,
1,1,1,1,0,1,1,1,0,1,1,0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,
0,1,1,1,1,0,0,1,1,1,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,0,0,1,0,
1,0,1,1,0,1,0,0,1,1,0,1,1,0,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,
1,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,1,0,0,0,1,0,1,0,0,
1,0,0,0,0,0,1,1,0,1,0,1,0,0,1,0,0,1,0,1,0,0,0,0,1,1,0,0,1,1,
1,1,0,1,0,0,1,1,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,1,1,0,0,1,0,
1,1,0,1,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,1,1,0,1,0,1,0,0,1,
1,0,0,1,0,1,1,1,0,0,0,0,1,1,1,0,0,1,1,0,1,1,0,0,0,0,0,1,0,0,
1,0,1,0,0,1,1,1,1,1,1,0,0,0,1,0,1,1,0,1,1,1,1,0,0,0,0,1,0,0,
1,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,0,
1,1,0,1,0,0,1,1,0,1,0,1,0,1,1,1,0,1,0,0,1,0,1,0,1,1,0,0,1,0,
0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,0,0,1,0,1,1,0,1,0,0,0,1,1,1,0,
1,1,0,1,0,0,1,0,0,0,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1,0,1,
0,0,1,1,1,1,1,1,0,0,1,1,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0,1,1,
1,1,1,0,1,1,0,0,1,0,0,1,0,1,1,1,1,0,0,1,1,1,1,0,1,1,1,0,1,0,
1,1,0,1,1,0,1,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,0,0,0,0,1,0,0,
1,0,0,0,1,1,1,0,0,0,1,1,0,0,1,1,1,0,1,1,1,1,1,1,1,0,0,0,1,1,
1,0,0,1,0,0,0,0,1,1,1,0,1,1,1,0,1,1,0,0,0,0,0,1,0,0,1,0,0,0,
1,0,0,0,0,1,0,1,1,1,0,0,1,0,1,1,1,1,0,1,0,1,1,0,0,0,0,0,0,1,
1,1,0,0,1,0,1,0,0,0,0,1,0,0,1,1,0,0,1,0,1,1,1,0,1,1,1,0,0,0,
1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0,
1,1,0,1,0,1,0,0,1,0,0,1,0,1,0,1,1,1,0,1,1,1,1,0,1,0,1,0,1,1,
1,1,1,0,0,0,1,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0,0,
1,0,0,1,1,0,1,0,0,1,0,0,1,1,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,
1,0,0,0,1,0,1,1,0,1,1,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,
0,1,1,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,0,0,1,
1,1,1,1,0,1,1,1,1,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,1,1,0,0,1,1,
1,1,0,1,0,0,1,1,0,1,1,1,0,0,0,0,1,0,1,1,0,1,1,1,1,1,1,1,1,1,
1,1,0,0,0,0,1,1,0,0,0,0,1,0,1,0,1,1,0,0,0,1,1,0,0,0,0,0,0,1,
1,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,0,0,0,
1,1,1,0,0,1,1,0,1,0,0,1,1,0,0,0,0,0,1,1,0,1,1,0,1,1,1,0,0,0,
1,0,0,1,0,0,1,0,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,0,1,1,0,1,1,1,
0,1,1,0,0,1,0,1,1,1,0,0,1,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1,
1,1,1,1,0,0,1,0,0,0,0,1,1,0,0,1,0,1,1,0,0,1,1,0,1,0,1,0,0,1,
1,1,1,1,0,1,1,0,1,0,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,1,1,0,1,0,
1,0,0,1,0,0,1,0,0,1,1,0,0,1,0,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,
1,1,0,1,0,1,0,0,1,0,0,1,0,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,1,0,
1,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,
1,0,1,1,0,0,0,1,1,0,0,1,0,1,1,1,1,0,1,0,0,0,0,1,1,0,1,1,1,0,
1,1,0,1,0,1,0,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,1,1,0,0,0,1,0,0,
1,0,0,1,1,1,0,0,0,1,1,1,1,1,1,0,1,0,1,1,1,0,0,0,1,0,0,0,0,0,
0,1,0,1,0,0,1,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,1,1,1,
1,1,0,1,0,0,1,1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,
0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,1,0,1,1,
1,0,1,0,1,1,0,1,1,1,0,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0,
1,0,0,1,0,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0,0,0,0,1,1,0,1,0,1,0,
1,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,0,1,0,0,1,1,1,1,1,1,0,1,1,1,
0,1,1,1,1,1,1,1,0,1,0,1,1,0,0,1,1,0,1,0,0,0,1,0,1,1,0,0,1,0,
1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,1,
1,1,0,1,1,1,0,0,0,0,1,1,0,0,1,1,0,0,0,1,0,1,1,1,1,0,0,1,1,0,
1,1,0,1,0,0,1,1,0,1,1,0,0,1,1,1,0,1,1,1,0,0,0,1,1,1,1,0,0,1,
1,0,0,1,0,1,1,0,0,1,1,1,0,0,1,0,0,1,0,1,0,1,1,0,1,0,1,1,0,0,
1,1,0,1,0,0,0,0,1,0,1,1,0,1,1,1,0,1,0,0,1,1,0,1,0,0,1,0,1,0,
0,0,1,1,1,0,0,1,1,0,0,1,0,1,0,1,1,0,0,0,1,1,1,0,1,0,0,0,1,1,
1,1,1,0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,
1,0,1,1,0,0,0,1,1,0,0,1,0,1,1,1,1,0,0,1,0,1,0,0,1,0,0,1,1,0,
1,1,0,1,0,0,0,0,1,0,0,1,0,1,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,0,
1,1,0,0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,
1,1,0,1,0,0,1,1,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,1,0,1,1,0,
0,1,0,1,1,0,1,0,0,0,1,0,0,1,0,1,0,0,1,1,1,1,0,1,1,1,1,0,0,1,
1,0,1,0,0,0,0,1,1,1,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,
1,0,1,0,0,1,0,1,1,1,0,0,1,1,1,1,1,1,0,0,0,1,1,0,0,1,0,0,0,0,
1,1,1,1,0,1,1,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,1,0,0,1,0,0,0,0,
1,1,1,0,0,0,1,1,1,0,0,0,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,0,1,0,
1,0,0,1,0,0,0,1,1,0,0,0,0,1,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,
1,1,0,1,0,0,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,
0,1,1,1,1,1,1,1,0,1,1,0,1,1,0,0,0,0,1,0,1,1,0,1,1,1,0,0,1,1,
1,1,1,0,0,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,0,1,1,
1,1,1,1,1,0,1,1,0,1,0,1,1,1,0,0,0,0,1,0,1,0,1,0,1,0,0,0,1,1,
0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,0,1,0,0,0,1,0,1,1,0,0,1,0,
1,0,1,0,0,1,1,0,0,0,0,1,1,1,1,0,0,1,1,1,0,1,1,0,1,1,1,0,0,0,
1,1,0,1,0,1,1,1,0,1,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,0,0,
1,1,0,1,0,0,1,1,0,1,1,1,0,1,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,
1,1,0,1,1,1,0,1,0,1,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,0,0,1,
1,1,1,1,0,0,1,0,0,0,0,1,1,0,0,0,1,1,1,1,0,1,1,0,1,0,1,0,1,1,
1,0,0,1,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,0,1,0,0,1,1,1,1,1,1,0,
1,0,0,0,1,1,0,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,
0,1,0,1,1,0,1,1,0,1,0,0,1,1,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,
1,0,0,0,0,0,1,0,0,0,0,1,1,0,1,0,1,1,1,1,1,1,0,0,1,1,0,0,0,1,
1,0,1,1,0,0,0,0,1,1,0,1,1,1,1,1,1,1,0,0,1,1,0,0,1,0,0,1,1,0,
1,1,0,0,0,1,0,0,1,0,0,1,0,1,1,0,1,1,1,1,1,0,1,1,1,0,0,0,1,1,
1,1,0,1,0,0,1,1,0,1,0,0,0,1,1,1,0,1,1,1,0,0,0,0,1,1,0,1,1,1,
1,0,0,0,0,0,1,0,0,0,1,1,1,0,1,0,1,1,1,1,0,0,0,1,1,1,1,1,1,0,
0,1,1,1,1,0,1,1,0,1,0,0,1,1,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,0,
1,1,0,1,0,0,1,1,0,1,1,0,0,1,1,0,0,1,0,1,0,0,0,1,1,1,1,0,0,1,
1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,1,1,1,0,1,1,0,0,1,0,
1,1,0,1,1,1,0,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,1,1,0,0,1,0,0,1,
1,1,0,1,0,0,1,1,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,1,1,1,1,0,1,
1,0,0,1,0,0,0,1,1,0,0,1,0,1,1,1,1,1,0,1,1,0,0,0,1,1,0,1,1,1,
1,1,0,1,0,0,1,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,0,0,1,0,1,1,
1,0,0,1,0,0,0,1,1,0,0,0,0,0,1,0,1,1,0,0,1,1,0,0,0,0,0,1,0,0,
1,1,1,1,0,1,1,1,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,
1,0,0,0,1,1,1,1,0,1,0,1,1,0,0,0,1,1,0,0,1,1,0,0,1,0,0,0,1,0,
1,1,1,0,1,1,1,0,0,0,0,1,0,1,0,1,0,0,1,1,0,1,1,0,1,1,1,0,0,0,
1,1,1,1,1,0,0,0,1,0,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,0,0,
1,1,1,1,1,1,0,0,1,0,1,1,0,0,0,1,1,0,0,1,0,0,1,1,1,1,0,1,0,1,
1,1,0,1,0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,
0,1,0,1,0,0,1,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,1,1,0,1,1,1,
1,1,1,1,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1,
0,1,0,1,1,1,0,1,0,1,0,0,1,1,0,0,1,0,1,1,1,0,1,1,0,0,1,0,0,1,
1,0,0,1,0,0,1,0,0,1,1,1,0,0,1,1,0,1,0,0,0,1,0,1,1,1,1,0,0,1,
0,1,0,1,1,1,0,0,0,0,0,0,0,1,1,0,1,0,1,1,0,1,1,1,0,0,0,0,0,1,
1,0,0,1,0,1,1,1,0,0,0,0,1,0,0,1,1,1,0,0,0,1,0,0,0,0,0,1,0,0,
1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,
1,1,1,1,0,1,1,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,1,0,1,1,0,
1,1,0,1,0,0,1,0,0,0,0,0,1,1,0,0,0,1,0,1,0,1,1,0,0,0,1,0,1,1,
1,0,1,1,0,0,0,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,1,0,1,0,1,0,1,
1,1,1,0,0,1,1,1,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0,
1,1,0,1,1,0,1,1,0,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,0,0,1,0,1,0,
1,0,0,1,0,0,0,1,1,0,0,0,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,1,0,0,
1,1,0,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,
1,0,0,1,0,0,0,0,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,1,1,0,1,0,
1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,
1,1,0,1,1,1,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,0,1,0,0,
1,1,1,1,0,0,1,1,0,1,0,0,0,1,1,0,0,1,1,1,1,0,1,0,0,1,0,0,0,0,
1,0,1,0,0,1,1,1,0,1,0,1,0,1,1,0,1,0,0,0,1,1,1,0,0,0,0,0,0,1,
1,0,0,0,0,0,0,1,1,1,1,0,1,0,1,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,
0,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,1,0,1,0,1,1,0,1,1,0,1,1,1,0,
1,1,1,1,0,0,1,1,1,1,0,1,0,0,1,1,0,1,1,0,0,0,1,0,1,0,0,0,1,1,
0,1,1,1,1,0,1,0,0,0,1,0,0,1,1,0,1,0,1,0,1,0,0,1,1,0,1,0,1,0,
1,0,0,1,1,0,0,1,0,0,0,1,0,1,1,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1,
1,1,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,1,1,0,1,1,1,0,1,1,0,0,1,0,
1,0,0,1,0,0,0,0,1,1,0,1,1,1,1,1,1,0,1,0,1,0,0,1,1,1,1,1,1,0,
1,0,0,1,0,0,0,0,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,
1,1,1,0,0,0,0,0,1,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,1,0,1,1,
1,0,1,0,0,0,0,1,1,1,1,0,1,0,1,1,1,0,1,0,1,0,0,0,0,1,0,1,0,0,
1,0,0,1,0,0,0,1,1,0,0,0,0,0,1,0,1,1,0,1,1,1,1,0,0,0,0,0,0,1,
1,1,0,1,0,0,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,
0,0,0,0,1,1,1,1,0,1,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,1,
1,1,0,1,1,0,1,0,0,0,0,1,0,0,1,1,0,0,1,0,1,1,1,0,1,1,1,0,0,0,
0,1,1,0,0,0,1,1,1,1,0,0,0,1,1,0,0,1,0,0,0,1,1,0,0,1,0,0,0,0,
1,1,0,1,0,0,1,1,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,
0,1,1,0,1,0,1,0,0,0,1,1,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,1,1,0,
1,1,0,1,0,0,0,0,1,0,0,1,0,0,1,1,1,1,0,0,1,1,0,0,0,0,0,1,0,0,
1,1,1,1,0,0,1,1,0,1,0,1,0,1,0,1,1,1,0,0,0,1,1,0,1,0,0,0,1,1,
1,0,0,1,0,0,1,1,0,0,1,0,1,1,0,0,0,1,0,1,0,1,1,1,0,0,1,1,1,0,
1,1,1,1,1,0,0,1,1,1,1,0,1,1,1,0,1,0,1,0,0,0,1,0,0,1,0,1,0,1,
1,1,0,1,1,0,1,1,0,1,0,0,1,1,1,0,0,1,1,0,1,0,0,0,1,1,0,1,1,1,
1,0,1,0,0,0,0,0,1,0,1,1,0,0,1,1,1,1,0,1,0,1,1,1,1,0,0,0,0,1,
0,1,1,0,1,0,1,0,0,0,0,1,0,1,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,1,
1,1,0,1,1,0,1,0,0,0,0,1,0,1,1,1,0,0,1,0,1,1,1,0,1,1,1,0,0,0,
1,1,0,1,0,1,1,1,0,1,0,0,0,0,1,0,1,1,1,0,0,1,1,0,0,0,0,0,0,1,
1,1,0,1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1,0,
1,0,1,0,0,0,0,1,1,1,0,1,1,1,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,0,
1,0,1,1,0,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,
1,1,1,1,1,0,0,1,1,1,1,0,1,1,0,0,1,0,1,1,0,0,0,0,1,0,0,0,1,0,
0,1,1,1,1,1,1,1,0,1,0,0,1,1,1,0,0,0,0,1,1,0,0,0,1,1,0,1,1,0,
1,1,1,0,0,0,0,0,1,0,1,1,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,1,
1,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,
0,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,
1,0,0,0,0,0,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,
1,0,1,0,1,0,0,1,1,1,0,0,1,0,1,0,0,0,1,1,1,0,0,1,0,1,0,0,0,1,
1,0,1,1,1,0,0,1,1,0,0,0,0,1,1,0,0,0,1,1,0,0,1,1,0,1,1,0,1,0,
1,1,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,0,0,1,
0,0,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,1,0,0,1,1,1,1,1,1,0,0,0,
1,1,1,1,1,0,0,1,1,1,0,0,0,1,1,0,1,0,0,0,1,0,0,0,1,0,0,1,1,0,
1,1,0,1,0,1,1,1,0,1,1,0,0,0,1,0,0,1,1,0,0,0,1,1,0,0,1,1,1,0,
1,0,0,1,0,0,0,0,1,1,0,1,1,0,1,0,0,1,1,1,0,0,0,1,1,1,1,1,0,1,
1,1,0,1,0,0,1,1,0,1,0,0,0,1,0,1,1,1,1,1,1,1,0,0,1,1,0,1,1,0,
1,0,0,1,0,0,1,0,0,1,0,0,0,1,1,0,0,1,1,0,1,0,0,0,0,0,0,1,0,0,
1,1,1,1,0,1,1,0,0,0,0,1,1,1,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,0,
1,1,0,1,0,0,1,1,0,1,0,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,
1,1,0,1,0,0,1,0,0,0,1,1,1,1,1,0,1,1,1,0,0,0,0,1,0,0,1,0,0,0,
1,1,0,1,1,0,1,1,0,1,0,0,1,1,1,0,0,1,1,0,0,1,0,0,1,1,0,1,1,0,
1,1,0,1,0,0,0,0,1,0,0,1,0,1,1,1,0,1,0,0,0,1,1,0,1,1,1,0,0,0,
1,1,0,1,0,0,1,0,0,0,0,1,1,0,0,1,0,1,1,0,0,1,1,0,1,1,1,0,0,0,
1,1,0,1,0,1,1,1,0,1,0,1,0,0,1,0,0,1,0,0,0,1,1,0,1,0,0,0,1,1,
1,0,0,1,1,0,0,1,0,0,0,1,0,1,1,0,1,1,0,0,0,0,0,0,1,1,0,1,0,1,
1,1,0,1,0,0,1,0,0,0,0,1,1,1,1,0,0,1,0,0,0,1,1,0,1,1,1,0,0,0,
1,1,0,0,0,0,0,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
1,1,0,1,0,0,1,1,0,1,1,0,0,1,1,1,0,0,1,0,1,1,0,1,1,1,1,0,0,1,
1,0,0,1,0,0,1,1,0,0,0,0,1,0,0,1,0,1,0,0,0,1,1,0,0,0,0,0,0,1,
1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,1,0,0,0,1,1,1,0,1,0,0,0,0,
1,1,0,1,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,0,1,0,1,1,1,0,1,0,
1,1,0,1,0,0,1,0,0,0,0,1,1,1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,0,0,
1,1,1,0,0,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,1,1,0,0,1,1,0,0,1,1,
1,0,1,1,0,1,1,1,0,0,1,0,1,0,1,0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,
1,0,0,1,0,1,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1,1,0,1,1,0,0,0,1,0,
1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,0,0,1,0,1,0,0,0,1,0,0,1,0,1,0,
1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,1,1,1,1,0,0,1,0,
1,1,0,1,0,0,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,0,
1,1,0,1,1,1,0,0,0,0,0,1,0,0,0,1,1,0,1,1,0,0,0,1,0,0,1,1,0,0,
0,1,0,1,1,1,1,0,0,0,1,1,0,1,1,1,1,0,1,1,0,0,1,1,1,0,0,1,1,0,
1,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,0,1,0,0,0,0,0,1,1,1,1,0,0,1,
1,1,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,1,0,0,1,0,0,1,0,0,1,1,0,
1,0,0,1,0,0,1,1,0,0,1,1,1,0,1,0,1,0,1,1,0,0,1,1,1,1,1,1,1,1,
1,1,0,1,0,0,1,0,0,0,0,1,1,1,0,0,0,1,1,0,0,1,1,0,1,1,1,0,0,0,
1,1,1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,1,0,0,1,0,0,0,0,1,0,1,0,1,
1,1,1,0,1,1,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,1,0,
1,1,1,0,1,1,1,1,0,1,1,1,1,0,0,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,
1,1,0,1,0,0,1,1,0,1,1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,0,1,0,1,0,
1,0,0,1,0,1,0,1,1,0,1,1,0,1,1,1,1,1,0,0,0,0,0,1,1,0,0,0,1,0,
0,1,1,1,1,0,1,1,0,1,0,1,1,1,0,1,1,0,1,1,1,0,0,0,0,1,0,1,0,0,
1,1,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,1,
1,0,1,0,0,0,0,1,1,1,0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,0,0,0,0,
1,1,1,1,0,0,1,0,1,0,0,1,1,0,1,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,
1,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,
1,0,0,1,0,1,0,1,1,0,0,1,0,1,1,1,1,1,0,0,1,1,1,0,1,0,0,0,1,1,
0,1,0,1,1,1,0,1,0,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0,1,0,0,1,1,0,
1,1,0,1,1,1,1,1,0,1,1,0,1,0,1,1,0,0,1,1,1,1,1,1,0,0,1,1,1,0,
1,1,0,1,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,1,0,1,1,1,0,0,1,1,0,0,
0,0,1,1,1,0,1,0,0,1,0,0,1,1,0,0,1,0,1,0,1,0,0,0,0,1,0,1,0,1,
1,1,1,1,0,0,1,0,1,0,0,1,1,1,1,0,0,1,0,0,0,0,1,0,1,0,1,0,0,1,
1,1,1,1,0,0,1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,1,0,
1,1,0,1,0,0,1,0,0,0,1,0,1,1,1,1,0,1,1,0,1,1,1,1,0,0,0,1,0,0,
1,1,0,0,0,0,1,0,0,0,1,1,1,0,0,1,0,1,0,0,1,1,1,1,1,1,0,1,1,1,
1,1,1,1,0,0,1,1,0,1,0,0,0,0,1,0,1,1,1,1,0,1,0,0,1,0,0,1,0,0,
1,1,1,1,0,1,0,1,1,1,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,1,
1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,0,1,0,0,0,1,0,0,
1,1,0,1,0,0,1,0,0,0,0,1,1,1,1,1,0,1,1,0,1,1,0,0,0,0,0,1,0,0,
1,1,0,1,0,0,1,1,0,1,1,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
1,1,1,1,0,0,1,0,0,0,0,1,1,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,0,1,
0,1,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,1,1,0,0,0,0,0,1,
1,1,1,1,0,1,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,0,1,1,1,1,0,0,1,
1,1,0,1,1,1,0,0,0,0,0,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,0,0,1,1,
1,1,0,1,1,0,1,0,0,0,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,1,1,0,1,0,
1,0,0,1,0,0,1,1,0,0,0,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0,0,0,0,1,
1,0,0,1,0,0,0,0,1,1,1,0,1,1,1,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,
1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,0,0,1,0,1,1,1,1,0,1,1,0,0,1,0,
1,1,0,1,0,0,1,1,0,1,0,0,0,1,1,1,0,1,0,0,0,0,0,0,1,1,0,1,1,1,
1,0,0,1,0,0,0,0,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,1,
1,1,0,1,0,0,1,0,0,0,1,1,1,1,0,1,0,1,1,0,1,0,0,1,0,0,1,0,1,0,
1,1,1,0,0,0,0,1,1,0,0,1,0,0,1,0,1,1,0,1,0,0,0,0,0,1,0,0,0,1,
0,1,1,1,0,0,1,1,0,1,0,1,0,1,1,0,0,0,1,1,0,0,0,0,0,1,0,1,0,1,
1,1,1,1,1,0,0,0,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,1,0,1,0,1,0,1,
1,1,1,1,0,0,0,0,1,0,0,1,0,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,0,1,
1,1,1,1,1,0,1,0,0,0,0,1,0,1,1,1,1,1,0,0,1,1,1,0,1,0,1,0,1,1,
1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,1,0,1,0,1,1,0,1,1,1,0,0,0,
0,1,1,1,0,1,1,0,1,0,0,0,1,1,0,0,0,0,0,1,1,0,1,0,0,0,1,0,1,1,
1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,
1,1,0,1,1,1,0,1,0,1,0,1,0,0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,1,1,
1,1,0,1,0,0,1,1,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,1,0,1,1,1,
1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,
1,1,0,1,1,1,0,1,0,1,0,0,0,1,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1,
1,0,0,1,0,0,1,1,0,0,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,1,0,0,0,0,
1,1,0,0,1,1,0,1,0,1,0,1,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,
1,1,0,1,0,0,1,0,0,0,1,1,1,1,0,1,0,1,0,0,0,1,1,1,1,1,0,1,1,1,
1,1,0,1,0,0,1,0,0,0,1,1,1,0,0,0,0,1,1,1,0,1,1,1,1,1,0,1,1,1,
1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,
1,1,0,1,0,0,1,0,0,0,0,1,1,1,1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,
1,0,1,1,0,1,1,1,1,0,0,0,1,0,0,0,0,1,0,1,0,1,1,0,0,1,0,0,0,0,
1,1,0,1,0,1,1,0,0,0,0,0,1,0,1,0,0,1,1,1,0,1,1,0,0,0,1,0,1,1,
1,1,1,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,0,1,0,1,1,0,1,1,1,0,0,0,
1,1,1,0,1,1,0,0,1,0,0,1,0,0,1,0,1,0,1,1,0,1,1,1,1,1,0,0,1,0,
1,1,0,1,0,1,0,1,1,1,0,0,1,1,0,0,1,1,0,1,1,1,1,0,0,0,0,0,0,1,
1,0,1,0,0,0,1,1,0,1,1,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,1,1,1,0,
1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,0,0,0,0,1,0,1,1,0,1,1,0,0,1,0,
1,0,1,0,1,1,1,0,0,0,1,1,0,1,1,1,0,0,1,1,1,0,1,1,1,1,0,1,1,1,
1,1,0,1,0,0,1,1,0,1,0,0,0,1,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,1,
1,0,1,1,0,1,0,1,1,0,0,0,0,1,1,0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,
1,1,1,1,0,1,1,0,1,0,1,1,1,0,1,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,
1,1,1,1,0,1,1,0,0,0,1,0,1,1,1,0,0,1,0,1,0,0,0,1,1,1,1,0,0,1,
1,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,1,1,0,1,0,1,1,1,0,1,0,
1,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,1,0,
1,1,0,1,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,0,1,0,1,
1,1,1,1,0,0,1,1,1,1,1,0,0,0,1,0,0,1,0,1,1,1,1,0,0,1,0,1,0,1,
1,1,1,1,0,0,1,0,0,0,1,1,1,1,0,0,0,0,1,0,1,0,0,1,0,1,1,0,1,1,
1,0,0,1,0,0,1,0,0,1,0,1,0,1,1,1,0,1,1,1,0,1,1,0,0,0,1,0,1,1,
1,0,0,1,0,0,0,0,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,0,0,0,1,0,0,1,
1,0,1,1,0,1,1,0,0,1,1,0,0,0,0,0,1,1,0,1,0,1,1,0,0,1,1,1,0,1,
1,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,
0,1,0,1,1,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1,1,0,0,1,1,1,1,0,0,1,
1,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,
1,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,1,1,0,0,1,1,1,0,
0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,1,1,1,0,0,1,1,1,0,0,1,0,0,1,
1,0,1,1,0,0,0,1,1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,
1,0,0,1,0,0,1,1,0,0,0,1,1,1,1,1,0,1,0,0,1,0,0,0,1,1,0,1,1,0,
1,0,1,1,0,1,1,1,1,0,0,0,1,1,1,0,0,1,1,0,1,0,1,0,0,1,0,0,0,0,
1,0,1,1,0,1,0,1,1,0,0,1,0,1,0,0,1,1,1,0,1,1,1,1,0,0,1,0,0,1,
1,1,1,0,0,0,0,1,1,1,1,0,1,1,1,0,1,1,0,1,0,0,0,1,1,0,1,1,1,0,
1,1,0,1,0,0,0,0,1,0,0,1,0,1,0,1,1,0,1,0,0,0,0,1,0,0,1,1,0,0,
1,1,0,1,0,0,1,1,0,1,0,1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,0,1,0,
1,1,0,1,1,0,1,0,0,0,0,1,0,1,1,1,0,0,1,1,1,0,1,0,1,1,1,0,0,0,
1,1,1,1,0,1,1,1,0,1,0,0,0,0,0,1,0,1,0,0,0,1,1,0,0,1,0,0,0,0,
0,1,0,1,0,0,1,0,0,0,0,0,1,1,0,1,0,1,0,1,0,1,1,0,0,0,1,0,1,1,
1,0,1,1,1,0,1,1,0,0,0,1,0,1,1,1,0,1,0,0,1,1,1,0,0,1,0,0,0,0,
1,0,0,0,0,0,0,1,1,1,0,0,1,1,1,0,1,1,0,0,1,0,1,0,0,0,0,0,0,1,
1,0,0,1,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,0,0,1,0,0,1,0,1,0,
1,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,0,1,0,1,0,0,1,0,0,0,0,1,0,0,
1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,0,1,1,0,1,0,0,0,
1,0,0,1,0,0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,0,0,0,1,1,1,0,0,1,1,
1,0,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,1,1,1,0,0,0,1,0,1,1,
1,1,0,1,0,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,0,0,1,0,0,0,0,0,
1,0,0,1,0,1,1,0,0,1,0,1,0,0,1,1,0,1,1,1,0,1,0,0,1,0,0,1,1,0,
1,0,1,1,0,0,0,1,1,0,1,0,0,1,1,0,1,1,0,0,0,0,0,1,0,1,1,0,0,1,
1,1,1,1,0,0,0,0,1,0,1,1,0,0,0,0,1,1,0,0,0,1,1,1,1,0,0,1,1,0,
1,1,0,1,0,1,1,1,0,1,0,0,0,0,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0,1,
1,1,0,1,0,0,1,1,0,1,0,0,0,0,1,0,0,1,0,1,1,0,0,0,1,1,0,1,1,0,
1,0,1,0,0,1,0,0,1,0,1,1,0,0,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,0,
1,0,0,1,0,0,0,1,1,0,1,1,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,0,1,
0,1,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,1,0,1,0,1,0,0,1,0,0,0,0,
1,0,0,0,0,0,0,0,1,0,0,1,0,1,1,0,1,1,0,0,1,0,0,0,1,1,0,0,1,1,
1,1,0,1,0,0,0,1,1,1,0,0,1,0,0,0,1,0,1,1,0,0,1,1,0,0,1,0,0,1,
1,1,1,1,0,1,1,1,0,1,1,0,0,1,1,0,1,1,0,1,0,0,0,1,1,1,1,0,1,1,
1,1,1,1,0,0,0,1,1,1,0,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,
1,0,1,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,1,0,0,1,1,0,1,0,0,0,0,
0,1,0,1,0,0,1,0,0,0,0,0,1,1,0,1,1,0,1,0,0,0,0,0,1,1,0,1,0,1,
0,1,0,1,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,1,0,0,0,1,0,1,1,
1,1,0,1,0,0,1,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,
1,0,0,1,0,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,0,0,0,0,0,1,0,0,
1,1,1,1,1,0,1,0,0,0,0,1,0,1,0,1,1,0,0,0,1,0,1,0,1,0,1,0,1,1,
1,1,1,1,1,0,1,0,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,1,0,1,
1,1,0,1,0,0,1,1,0,1,1,1,0,1,1,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,
1,0,0,1,0,0,0,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,0,0,1,0,0,1,
1,1,0,1,0,0,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0,0,1,0,1,1,0,0,1,0,
1,1,0,1,0,1,0,1,1,1,0,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,
1,1,1,0,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,
1,0,1,1,1,0,0,0,1,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,0,0,1,0,0,
0,1,1,1,1,1,0,1,1,1,0,0,0,0,1,1,1,1,0,1,0,0,0,0,1,1,0,1,1,1,
1,0,0,1,0,0,1,1,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,
1,1,0,1,0,0,1,1,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,
1,1,0,1,0,0,1,0,0,0,0,1,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,
1,0,0,0,1,1,0,1,0,1,0,0,0,1,1,0,1,0,1,0,1,1,1,1,0,0,1,0,0,1,
1,1,0,1,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,1,1,1,1,1,1,0,1,1,1,
1,1,0,1,0,0,1,1,0,1,0,0,0,1,0,0,0,1,1,1,0,0,0,0,1,1,0,1,1,0,
1,0,0,1,0,0,0,1,1,0,0,1,0,1,0,1,0,1,0,0,0,1,0,0,1,1,0,1,1,0,
1,1,1,1,0,1,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,1,
1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,0,0,0,0,1,
1,1,0,1,0,0,1,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,
1,1,1,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,
1,1,1,1,0,1,0,0,1,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,
1,1,1,1,0,0,1,0,1,0,0,0,1,1,1,0,0,1,1,0,1,0,1,0,0,1,1,0,1,0,
1,1,1,0,0,0,1,0,1,0,0,1,1,0,0,0,0,1,0,1,1,0,0,0,0,1,0,0,0,1,
0,0,1,1,1,0,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,0,1,1,1,1,1,
0,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,0,1,1,1,1,0,0,1,
0,0,0,1,1,0,0,0,0,1,0,1,0,0,1,1,1,1,0,0,1,1,1,0,1,1,1,0,1,0,
1,0,0,1,0,0,0,1,1,0,0,1,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,1,0,
1,1,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,0,0,
1,0,0,1,0,0,1,1,0,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,
0,0,0,1,1,1,0,0,0,1,0,1,1,0,0,1,1,0,0,1,0,1,1,0,0,0,1,0,0,1,
1,0,0,1,0,0,0,0,1,1,0,1,1,1,0,0,1,1,0,1,1,0,1,0,0,0,1,0,0,1,
1,0,0,1,0,0,0,0,1,1,0,1,1,0,1,1,1,1,0,0,1,1,0,0,1,1,0,1,0,1,
1,0,1,1,1,0,0,1,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,1,1,1,0,1,
1,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,1,1,1,0,0,0,1,0,0,0,1,1,0,0,
0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,1,1,0,0,1,1,0,1,1,1,0,
1,0,0,1,1,1,0,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,0,1,0,1,1,
1,0,1,1,0,0,0,0,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,0,0,1,1,0,0,0,
1,1,0,1,1,0,0,1,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,1,0,0,1,0,0,0,
1,1,0,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,0,1,0,0,0,0,1,1,0,1,0,0,
1,1,1,1,1,0,1,1,0,1,0,0,1,1,0,0,0,0,1,1,0,1,1,0,0,1,0,0,0,0,
1,1,0,1,0,1,0,0,1,0,1,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,1,0,0,0,
1,0,1,1,0,1,1,0,1,1,1,0,0,0,1,0,1,1,0,1,1,0,1,0,0,1,1,1,0,1,
1,1,1,1,0,0,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,0,0,1,0,1,0,0,
1,1,0,1,1,1,0,1,0,1,0,1,0,0,1,0,0,0,0,1,0,1,1,0,1,0,0,0,1,1,
1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,0,1,0,1,1,0,1,0,1,0,0,0,1,1,
1,0,1,0,1,0,1,0,0,0,0,1,0,1,1,1,0,0,1,0,1,0,1,0,1,0,1,0,0,1,
1,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,0,1,0,0,0,0,1,0,1,0,0,1,1,0,
1,1,0,1,0,0,1,1,0,1,0,1,0,1,1,0,0,1,1,0,0,0,1,0,1,1,0,0,1,0,
1,0,0,0,1,0,0,1,0,1,0,0,1,1,1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,
0,1,0,1,1,0,1,1,0,1,1,1,1,1,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,
0,0,0,1,0,0,0,1,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,
1,1,1,1,0,0,1,1,1,1,0,1,0,0,1,1,0,1,1,1,1,0,0,0,0,1,0,1,0,1,
1,1,1,1,0,0,1,1,0,1,0,1,0,1,0,1,0,0,0,0,0,1,1,0,1,0,0,0,1,1,
1,1,0,1,1,1,1,1,0,1,0,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,1,0,0,
1,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0,1,1,0,0,0,0,
1,1,1,0,1,0,0,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,0,1,1,0,0,0,
1,1,1,1,1,1,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
1,0,0,0,0,1,0,0,1,0,0,1,0,1,1,0,1,1,0,1,1,0,0,0,1,0,0,0,0,0,
1,1,0,1,0,1,1,1,0,1,0,1,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,1,0,0,
1,1,0,1,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,0,1,1,1,1,0,1,
1,1,0,1,0,0,1,0,0,0,1,1,1,1,1,1,0,1,0,0,1,1,1,0,1,1,1,1,0,1,
1,1,1,1,0,1,0,0,1,0,1,1,0,0,1,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,
1,1,0,1,0,0,0,0,1,0,0,1,0,1,1,1,1,1,0,0,0,1,1,0,1,1,1,0,1,0,
1,1,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,0,0,1,
1,0,0,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,1,1,1,0,0,1,1,0,0,
0,1,0,1,1,0,1,0,0,0,1,0,0,0,1,0,0,1,1,0,0,0,0,1,1,1,1,0,0,1,
1,0,1,1,0,0,0,1,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,
1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,0,1,0,0,0,0,1,0,1,0,1,0,0,1,
1,1,1,0,0,1,1,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,1,1,
1,1,0,1,1,1,1,1,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,1,0,
1,1,1,1,0,0,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,1,0,1,0,0,0,
1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,1,0,1,1,0,1,1,0,1,0,1,1,1,0,
0,1,0,1,1,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,1,1,0,0,0,1,0,0,
1,1,0,1,1,0,0,1,0,1,0,0,1,1,1,0,1,1,0,1,1,1,0,0,1,1,0,1,1,0,
1,0,0,0,1,1,1,1,0,1,0,0,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,
1,0,0,1,0,0,1,0,0,1,0,1,0,1,1,1,1,0,0,1,0,0,0,0,1,1,0,1,1,1,
1,1,0,1,0,0,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,
0,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,1,1,1,0,1,0,0,0,1,0,0,1,
1,0,0,1,0,0,1,1,0,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,
1,1,1,1,0,0,1,0,0,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,
1,0,0,0,1,0,1,1,0,1,1,1,1,0,1,1,0,0,0,0,0,1,0,0,1,1,0,1,1,0,
1,0,0,0,1,1,0,0,0,0,0,1,0,1,0,1,1,0,0,1,1,0,0,0,1,0,0,0,1,0,
1,0,0,1,0,1,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,1,0,0,
0,0,0,0,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,0,1,0,1,1,0,0,1,0,0,1,
1,0,0,1,0,0,0,0,1,1,0,1,1,0,1,1,1,1,0,0,1,1,1,0,1,1,1,0,1,0,
1,0,0,1,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,1,0,1,0,1,0,
1,1,1,1,0,1,1,0,0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,0,0,1,1,0,1,0,
1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,0,0,1,1,0,1,0,0,0,1,1,
1,0,0,1,0,0,1,1,0,0,0,1,1,1,1,1,0,1,0,0,1,1,1,0,0,0,0,0,0,1,
1,1,1,1,0,1,0,1,1,1,0,1,1,0,0,0,1,1,0,0,0,1,1,0,1,1,0,0,1,0,
1,0,0,1,1,0,1,0,0,1,0,0,1,1,1,0,0,0,1,0,1,1,0,0,0,0,0,1,0,0,
1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,1,1,0,0,0,
1,1,0,1,0,0,1,1,0,1,0,0,0,1,0,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,
1,1,1,1,0,0,0,0,1,0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,1,1,1,0,1,
1,0,1,1,0,0,0,1,1,0,1,1,0,0,1,0,1,1,0,1,1,1,1,1,1,0,1,1,1,0,
1,0,0,0,0,0,0,0,1,0,1,1,0,0,1,1,1,0,1,0,0,1,1,0,0,0,1,1,0,0,
1,0,1,0,1,1,1,1,0,1,0,0,1,1,1,0,1,0,1,0,0,1,1,0,0,1,0,0,0,0,
0,0,1,1,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,1,0,0,0,0,
1,1,1,1,0,1,1,0,1,0,0,1,1,0,1,1,1,1,1,1,0,0,1,0,1,1,1,0,1,0,
1,0,1,1,0,0,1,0,1,1,1,1,0,1,1,1,0,0,1,0,0,1,1,1,0,1,0,1,0,1,
1,1,0,1,0,0,1,1,0,1,1,0,0,1,0,0,0,1,0,0,1,1,1,0,0,0,0,1,0,0,
1,0,1,0,0,1,0,0,1,0,0,1,0,1,1,0,1,1,1,1,1,1,0,1,1,1,1,0,1,1,
1,1,1,1,1,0,1,0,0,0,0,1,0,1,1,1,0,1,0,0,1,1,1,0,1,0,1,0,0,1,
0,1,1,1,0,1,0,1,1,1,0,0,1,1,0,0,1,0,1,0,1,1,1,1,0,1,1,0,0,0,
1,0,1,1,0,0,0,0,1,1,0,1,1,1,1,1,1,1,0,0,1,1,0,0,1,0,0,1,0,0,
1,1,0,1,0,0,1,0,0,0,1,0,1,1,0,0,1,1,0,0,1,0,0,0,1,1,0,0,0,1,
1,1,0,1,0,0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,
1,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,1,0,0,0,1,1,0,1,1,1,0,0,0,
1,1,0,1,1,0,1,1,0,1,0,0,1,1,1,0,0,1,0,1,1,0,0,0,1,1,0,1,1,0,
1,1,0,1,0,1,1,1,0,1,1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,
0,1,1,1,1,1,1,1,0,1,1,0,1,0,0,1,0,0,1,1,1,1,0,1,1,1,1,0,0,1,
1,0,0,1,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,0,0,0,0,
1,1,1,1,1,1,1,1,0,1,1,0,1,0,0,0,0,0,1,1,1,0,1,1,0,0,1,1,1,0,
1,0,0,1,0,1,1,1,0,0,0,0,1,1,1,1,1,0,1,1,0,1,1,0,0,0,0,0,0,1,
1,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,
1,1,0,1,0,0,0,1,1,1,0,1,1,0,0,0,0,1,0,0,1,0,1,0,1,1,0,0,1,0,
1,0,0,1,1,1,1,1,0,0,0,1,0,0,1,1,1,1,0,1,1,0,1,0,1,0,0,0,0,1,
1,1,1,1,0,0,1,1,0,1,1,1,0,0,0,1,0,1,0,0,0,1,1,0,1,0,0,1,1,0,
1,0,1,0,0,0,1,0,1,0,1,1,1,1,1,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,
0,0,0,1,0,0,0,0,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,0,0,0,1,1,0,0,
1,0,0,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,
1,1,1,1,0,0,1,1,0,1,1,0,0,1,1,1,0,1,1,1,0,0,0,1,1,0,1,0,0,0,
0,1,1,1,1,0,1,1,0,1,1,0,1,1,0,0,0,1,0,0,1,0,1,1,0,1,1,1,1,1,
0,1,0,1,0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,1,1,1,0,
1,0,1,1,0,1,1,0,0,1,0,1,0,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,
1,0,1,0,1,1,1,1,0,1,0,0,1,0,1,0,1,1,1,0,1,1,1,0,0,1,0,0,0,0,
1,1,0,1,0,1,0,0,1,0,1,1,0,0,1,1,1,1,0,1,0,1,1,1,1,0,0,1,1,0,
1,1,0,1,0,0,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,1,1,
1,1,0,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,1,0,0,0,1,1,1,1,0,1,0,1,
1,0,0,1,1,1,0,1,0,0,0,1,0,1,1,1,0,1,1,0,0,1,1,1,0,0,1,0,1,1,
1,0,0,1,0,0,0,0,1,1,0,0,1,0,1,1,0,0,0,0,1,1,1,0,0,0,1,0,1,1,
0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,0,0,
1,0,0,1,0,0,0,1,1,0,1,0,0,1,1,1,1,1,1,0,1,0,1,0,0,0,0,1,0,0,
1,1,1,0,1,1,1,1,0,0,0,1,0,1,1,0,0,1,0,0,1,1,1,0,1,1,0,0,1,0,
1,0,0,1,0,0,1,1,0,0,0,0,1,1,1,0,0,1,0,0,0,1,1,0,0,0,0,0,0,1,
1,0,1,1,1,1,1,1,0,0,0,1,0,1,1,0,0,0,1,0,1,1,1,0,0,0,0,0,0,1,
1,0,0,1,0,1,0,0,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,0,0,1,0,0,1,
1,1,0,0,0,0,1,1,0,0,0,0,1,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,
1,1,1,0,0,0,0,1,1,1,0,0,1,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0,1,0,
0,1,1,1,1,1,1,0,0,0,1,1,0,1,1,0,1,1,1,0,0,1,1,1,1,1,0,1,0,1,
1,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,1,1,1,0,1,1,
0,1,1,1,1,0,1,0,0,0,0,0,0,1,0,1,0,0,1,1,1,1,1,0,0,1,1,0,1,0,
1,0,0,0,1,0,1,0,0,0,0,1,0,1,1,1,0,1,1,0,1,1,1,0,0,0,1,0,1,1,
1,0,1,1,0,1,1,1,0,0,1,0,1,0,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,
1,1,1,1,0,1,1,0,1,0,1,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,0,1,1,1,
1,0,1,1,0,1,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,0,1,0,1,1,0,0,1,0,
1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,
1,0,0,1,0,0,0,1,1,0,0,1,0,0,1,1,0,1,0,0,1,1,1,0,0,0,0,0,0,1,
1,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,0,
1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,1,1,
1,1,1,1,0,0,1,0,0,0,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,1,0,0,
1,1,0,1,0,0,1,1,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,1,0,1,1,1,
1,1,1,1,0,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,1,0,
1,1,0,0,0,1,1,1,0,1,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,1,
1,1,1,1,0,0,1,0,0,0,0,1,1,1,1,1,0,1,0,0,1,0,0,0,0,1,0,1,0,1,
1,0,0,1,0,0,0,1,1,0,0,0,0,0,1,0,1,1,0,0,1,0,1,0,0,0,0,0,0,1,
1,1,1,1,1,1,1,0,0,0,0,1,0,1,1,1,0,0,1,1,1,1,1,0,1,1,1,0,0,0,
1,1,0,1,1,1,0,0,0,0,0,0,0,0,1,0,1,1,0,1,1,0,0,0,1,0,0,1,1,0,
1,0,1,1,1,1,0,1,1,0,0,1,0,0,1,1,1,1,0,1,0,1,1,0,0,0,0,0,0,1,
1,0,0,1,0,0,0,1,1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,
0,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,0,1,1,1,1,0,0,1,
0,0,1,0,0,0,1,0,0,0,1,1,1,0,1,1,0,1,1,0,0,1,1,1,1,0,0,0,1,1,
0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,1,
1,1,0,1,1,1,0,1,0,1,0,1,0,0,1,1,1,1,0,0,0,1,1,0,1,0,0,0,1,1,
1,0,0,1,0,1,1,1,0,0,1,1,1,1,1,1,0,1,0,1,0,0,0,1,1,0,1,0,0,0,
1,0,0,0,0,0,1,0,0,0,0,1,1,1,1,0,0,1,1,0,1,0,0,0,1,1,0,0,1,1,
1,1,1,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,0,
1,1,1,0,0,0,0,0,1,0,0,1,0,1,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,
1,0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,0,0,0,1,0,0,
1,0,0,1,1,0,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,
0,1,0,1,0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,
0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,
0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,1,0,0,1,1,1,1,1,0,0,0,1,1,1,0,
1,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,
1,1,0,1,0,0,1,1,0,1,0,0,0,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,1,
1,0,0,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,1,0,1,1,0,0,0,1,0,0,1,
1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,
1,1,1,0,0,1,0,0,1,0,0,1,0,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,
1,0,0,1,1,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,1,1,0,0,1,1,1,0,
1,0,0,1,0,0,0,1,1,0,1,1,0,1,1,1,1,1,0,1,1,0,0,1,1,1,1,0,1,1,
0,1,1,1,0,0,1,1,1,1,0,1,0,1,0,1,1,0,1,0,1,1,0,0,0,1,0,1,0,1,
1,0,0,1,0,0,0,0,1,1,1,1,1,1,1,0,1,1,0,0,0,1,1,1,0,0,0,1,0,0,
1,0,1,1,1,0,0,1,1,0,0,0,0,1,1,0,1,0,1,1,0,0,1,1,0,1,1,0,0,0,
1,0,0,1,0,0,0,0,1,1,0,1,1,0,1,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,
1,0,1,1,0,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,
1,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,1,0,1,1,
1,1,0,1,0,0,1,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,1,0,1,1,0,0,1,0,
1,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,
1,0,0,0,0,1,1,1,0,1,0,1,0,1,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,
1,0,0,1,0,0,1,0,0,1,0,1,0,0,1,1,0,1,0,0,0,1,1,0,0,0,1,0,1,1,
1,0,1,0,1,0,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,
1,0,0,1,1,1,1,1,0,0,1,0,0,1,1,1,0,0,1,0,1,0,1,1,0,0,1,1,1,0,
1,0,0,0,0,0,1,1,0,1,1,1,0,0,1,1,0,1,1,1,0,0,0,1,1,1,1,1,0,1,
1,1,0,1,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,1,1,0,1,1,0,
1,0,0,1,0,0,1,0,0,1,1,1,0,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,
0,0,0,1,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,1,1,1,1,0,0,0,0,0,0,1,
1,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,1,0,0,0,0,0,0,1,
0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,0,
1,1,0,1,1,0,1,0,0,0,1,1,0,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,
1,0,0,1,0,1,1,1,0,0,0,0,1,1,1,0,0,1,0,0,0,1,1,0,0,0,0,0,0,1,
1,0,0,1,0,0,0,0,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,1,0,0,0,0,0,1,
1,0,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,1,1,0,0,1,0,1,0,1,0,0,1,
1,0,0,1,0,0,1,1,0,0,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,0,0,1,0,0,
1,1,0,1,0,1,1,1,0,1,0,1,0,0,1,1,0,1,0,0,1,0,1,0,1,0,0,0,1,1,
1,0,0,0,0,0,1,1,0,1,0,0,0,0,1,0,1,0,1,1,0,1,1,0,0,0,0,0,0,1,
1,1,0,1,0,0,0,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
1,0,1,1,0,0,1,0,0,1,0,0,0,1,1,0,0,1,1,0,1,0,0,0,0,1,0,1,0,0,
1,1,0,1,1,0,1,0,0,0,0,0,0,1,1,1,0,1,0,1,1,1,1,0,0,0,1,0,1,1,
1,1,0,1,0,0,1,0,0,0,1,1,1,1,1,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,
1,1,0,1,0,0,1,0,0,0,0,0,1,0,1,0,1,1,0,1,0,0,0,0,1,1,0,1,0,0,
1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,
1,0,1,0,0,1,0,1,1,1,0,0,1,1,1,0,0,0,1,0,1,1,0,1,0,0,0,0,0,0,
1,1,1,0,1,1,1,0,0,0,1,1,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,
1,0,1,1,1,1,1,0,0,1,1,0,1,1,1,0,0,0,1,0,1,0,0,1,0,1,1,0,1,1,
1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,0,0,1,0,0,1,1,0,0,0,0,0,1,0,0,
1,0,1,1,0,1,1,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,1,1,0,1,1,0,
1,1,0,1,0,0,1,0,0,0,0,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,
0,0,0,1,0,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
1,0,1,1,1,0,1,0,0,1,0,1,1,1,1,1,0,1,0,0,0,1,1,0,1,0,1,0,0,1,
1,0,0,1,0,0,1,0,0,1,0,1,0,1,1,0,0,1,0,0,1,0,1,0,1,1,1,0,0,0,
0,1,1,1,0,0,0,1,1,1,1,0,1,1,0,0,1,0,1,1,0,1,1,0,0,1,0,1,0,1,
0,0,0,1,1,1,1,0,0,1,1,0,1,0,0,0,0,0,1,1,0,1,1,1,0,0,0,1,0,0,
1,1,0,1,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1,
1,1,0,1,0,0,1,1,0,1,1,0,0,1,1,0,0,0,1,1,0,1,1,1,0,0,1,1,1,0,
1,1,1,0,0,0,1,0,1,0,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,1,1,1,1,0,
1,1,1,0,0,1,0,1,1,0,0,0,0,1,0,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,
1,0,0,1,1,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,1,1,1,0,0,0,0,0,0,1,
1,1,1,1,0,0,1,0,1,0,0,1,1,0,0,0,0,1,1,0,0,0,1,0,1,0,1,0,0,1,
1,0,0,1,0,0,0,0,1,1,0,1,1,1,1,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,
1,0,1,0,0,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,
1,1,0,1,1,0,0,0,0,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,0,0,1,0,1,0,
1,0,0,1,0,0,0,0,1,1,0,0,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,
1,0,0,1,0,0,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,0,0,1,1,0,0,
1,1,0,1,1,1,1,1,0,1,1,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,1,1,1,0,
1,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,1,0,1,1,1,0,1,0,0,0,1,1,0,0,
1,1,0,1,0,0,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
1,0,0,0,0,0,0,1,1,1,0,0,1,0,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,1,
1,1,1,1,0,1,1,1,0,1,1,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,
1,1,0,1,0,1,1,0,0,0,1,1,1,0,1,1,0,1,0,0,0,1,1,0,1,0,1,1,0,0,
1,0,0,0,0,0,0,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
1,0,0,1,0,0,0,0,1,1,1,0,1,1,1,0,1,1,0,0,1,0,1,1,0,0,0,1,0,0,
0,1,0,1,0,0,1,1,0,1,1,1,0,1,0,1,1,0,0,1,0,1,0,1,0,0,1,0,0,0,
1,1,1,0,0,1,1,1,0,0,0,0,1,0,1,1,0,0,1,1,0,1,1,0,0,0,0,0,0,1,
1,1,0,1,1,0,1,0,0,0,1,1,0,1,1,0,1,0,1,0,1,1,1,1,1,1,0,1,1,1,
1,1,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,1,1,1,0,1,0,1,0,1,
1,0,0,1,0,0,0,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,0,0,1,1,0,1,0,1,
1,0,0,1,0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,1,0,1,1,0,0,0,0,0,0,1,
1,0,0,0,1,1,0,0,0,0,0,1,0,1,0,1,1,1,0,1,0,1,1,0,1,0,1,0,1,1,
1,1,0,1,1,1,1,0,0,0,1,1,0,1,0,0,0,0,1,1,0,1,1,1,1,0,0,1,1,0,
1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,0,1,0,0,1,0,0,1,0,0,0,
1,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,1,1,0,1,1,0,1,0,0,0,1,0,0,1,
1,1,1,0,0,1,0,1,1,1,0,1,1,0,0,1,1,0,1,0,1,0,0,1,0,1,1,0,0,1,
1,1,0,1,0,0,1,0,0,0,1,0,1,1,0,1,0,0,1,0,0,0,0,1,1,1,1,0,0,1,
1,0,0,0,0,0,1,0,0,0,1,1,1,0,1,1,0,1,0,0,0,0,0,1,1,1,1,1,0,0,
1,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,1,1,0,1,1,0,1,0,0,0,0,0,0,1,
1,1,0,1,0,0,1,0,0,0,0,1,1,1,1,1,0,1,0,1,1,0,0,0,0,0,0,1,0,0,
1,1,0,1,0,0,1,0,0,0,1,1,1,1,0,1,1,1,0,1,0,1,1,0,1,1,1,1,1,1,
0,0,0,1,1,0,1,1,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,1,1,0,1,1,0,
1,1,0,1,0,0,1,1,0,1,0,0,0,0,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,1,
1,1,0,1,0,0,1,1,0,1,1,1,0,1,0,0,0,1,0,0,0,1,1,0,1,1,0,1,1,1,
1,0,0,0,1,1,0,0,0,0,0,1,0,1,1,1,1,0,1,1,0,1,1,1,1,0,0,0,1,1,
1,0,1,0,0,0,0,0,1,0,0,1,0,1,1,0,1,1,0,0,1,0,0,0,1,0,0,0,1,0,
1,1,1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,0,1,0,1,0,0,1,0,1,1,1,1,1,
0,1,1,1,1,0,1,0,0,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0,0,1,1,0,1,0,
1,1,0,1,0,0,1,0,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,0,0,0,1,0,1,1,
1,1,0,1,0,1,1,0,0,0,0,1,1,0,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,
1,1,0,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1,0,1,1,0,0,1,0,0,1,
1,1,1,1,0,1,1,1,1,1,1,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,
1,1,1,1,0,1,0,0,1,0,0,1,0,0,0,1,1,0,1,1,1,0,1,1,1,1,0,0,0,0,
1,1,0,1,1,1,0,0,0,0,1,0,0,1,1,1,1,0,1,1,1,1,0,0,1,0,0,0,1,0,
1,1,1,1,0,1,1,0,0,0,1,0,1,1,0,0,1,1,1,1,0,0,1,1,0,1,0,1,0,1,
1,0,0,1,0,0,1,1,0,0,0,1,1,1,1,0,1,1,1,0,1,0,1,0,0,0,0,0,0,1,
0,0,0,1,1,0,0,1,0,0,0,1,0,1,1,0,1,1,0,0,1,1,1,0,1,1,0,0,1,0,
1,1,0,1,0,0,1,0,0,0,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,
1,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0,0,0,0,0,1,
0,0,0,0,1,1,0,0,0,0,0,1,0,0,1,0,1,0,1,1,0,1,1,1,1,0,0,0,1,1,
1,1,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,1,0,0,0,0,0,1,1,1,0,0,1,1,
1,1,0,1,0,0,0,0,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,
}

#endif

