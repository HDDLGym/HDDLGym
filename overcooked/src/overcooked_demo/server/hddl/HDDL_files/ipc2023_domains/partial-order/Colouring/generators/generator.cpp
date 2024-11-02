#include <bits/stdc++.h>

using namespace std;

const int oo = 0x3f3f3f3f;
const double eps = 1e-9;
const double PI = 2.0 * acos(0.0);


typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<string> vs;

#define sz(c) int((c).size())
#define all(c) (c).begin(), (c).end()
#define FOR(i,a,b) for (int i = (a); i < (b); i++)
#define FORS(i,a,b,s) for (int i = (a); i < (b); i=i+(s))
#define FORD(i,a,b) for (int i = int(b)-1; i >= (a); i--)
#define FORIT(i,c) for (__typeof__((c).begin()) i = (c).begin(); i != (c).end(); i++)



int arr[100][100][4];

int UP = 0;
int DOWN = 1;
int LEFT = 2;
int RIGHT = 3;

int main(int argc, char *argv[]){
	int n = atoi(argv[1]); // size of the field
	int s = atoi(argv[2]); // seed
	int p = atoi(argv[3]); // problem number
	int c = atoi(argv[4]); // colours

	srand(s);

	arr[0][0][LEFT] = rand()%c;
	FOR(j,0,n)
		arr[0][j][DOWN] = rand() % c;

	FOR(i,0,n){
		FOR(j,0,n){
			// select upper and right
			arr[i][j][UP] = rand() % c;
			arr[i][j][RIGHT] = rand() % c;

			arr[i+1][j][LEFT] = arr[i][j][RIGHT];
			arr[i][j+1][DOWN] = arr[i][j][UP];
		}	
	}	

	map<vector<int>, int> tiles;

	FOR(i,0,n){
		FOR(j,0,n){
			//cout << arr[i][j][UP] << " " << arr[i][j][DOWN] << " " << arr[i][j][LEFT] << " " << arr[i][j][RIGHT] << " " << endl;

			vi v;
			v.push_back(arr[i][j][UP]);
			v.push_back(arr[i][j][DOWN]);
			v.push_back(arr[i][j][LEFT]);
			v.push_back(arr[i][j][RIGHT]);

			tiles[v]++;
		}
	}
	int maxN = n;

	for(auto [v,i] : tiles)
		if (i > maxN) maxN = i;



	cout << "(define (problem tiling-" << p << ")" << endl;
	cout << "  (:domain game)" << endl;
	cout << "  (:objects" << endl;
	cout << "   ";
	FOR(i,0,maxN+1) cout << " n" << i;
	cout << " - number" << endl; 
	FOR(i,0,c) cout << " c" << i;
	cout << " IGN - colour" << endl; 
	cout << "  )" << endl;
	cout << "  (:init" << endl;
	FOR(i,0,maxN)
		cout << "    (next n" << i << " n" << i+1 << ")" << endl;
	cout << "    (max n" << n << ")" << endl;
	cout << "    (min n0)" << endl;
	cout << "    (cur_colour_next IGN)" << endl;
	cout << "    (isIGN IGN)" << endl;
	FOR(i,0,c) cout << "    (colok IGN c" << i << ")" << endl;
	FOR(i,0,c) cout << "    (colok c" << i << " c" << i << ")" << endl;
	for(auto [v,i] : tiles){
		cout << "    (tile c" << v[0] << " c" << v[1] << " c" << v[2] << " c" << v[3] << " n" << i << ")" << endl;
	}
	cout << "  )" << endl;

	cout << "  (:htn" << endl;
	cout << "    :tasks (and" << endl;
	cout << "      (rows n0)" << endl;
	FOR(i,0,n)
		cout << "      (start_line n" << i << ")" << endl;
	cout << "    )" << endl;
	cout << "  )" << endl;

	cout << ")" << endl;
}
