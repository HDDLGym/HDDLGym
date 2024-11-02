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


bool arr[100][100];

int main(int argc, char *argv[]){
	int n = atoi(argv[1]); // size of the field
	int s = atoi(argv[2]); // seed
	int p = atoi(argv[3]); // problem number
	int pl = atoi(argv[4]); // number of rounds

	srand(s);

	cout << "(define (problem game-" << p << ")" << endl;
	cout << "  (:domain game)" << endl;
	cout << "  (:objects" << endl;
	cout << "   ";
	FOR(i,0,n) cout << " l" << i;
	cout << " - coord" << endl; 
	cout << "  )" << endl;
	cout << "  (:init" << endl;
	FOR(i,0,n) FOR(j,0,n) {
		bool x = rand() % 2;
		arr[i][j] = x;
		cout << "    (on l" << i << " l" << j << " " << (x?"Y":"N")<< ")" << endl;
	}
	FOR(i,0,n-1)
		cout << "    (next l" << i << " l" << i+1 << ")" << endl;
	cout << "    (max l" << n-1 << ")" << endl;
	cout << "    (min l0)" << endl;
	cout << "  )" << endl;

	cout << "  (:htn" << endl;
	cout << "    :tasks (and" << endl;
	cout << "      (play)" << endl;
	cout << "    )" << endl;
	cout << "  )" << endl;



	FOR(rr,0,pl){
		int X = rand() % n;
		int Y = rand() % n;
		bool state = arr[X][Y];

		arr[X][Y] = !state;
		int x = X; int y = Y;
		x--;
		while (x >= 0){
			if (arr[x][y] != state) break;
			arr[x][y] = !state;
			x--;
		}

		x = X; y = Y;
		x++;
		while (x < n){
			if (arr[x][y] != state) break;
			arr[x][y] = !state;
			x++;
		}

		x = X; y = Y;
		y--;
		while (y >= 0){
			if (arr[x][y] != state) break;
			arr[x][y] = !state;
			y--;
		}

		x = X; y = Y;
		y++;
		while (y < n){
			if (arr[x][y] != state) break;
			arr[x][y] = !state;
			y++;
		}
	}



	cout << "  (:goal (and" << endl;
	FOR(i,0,n) FOR(j,0,n) {
		cout << "    (on l" << i << " l" << j << " " << (arr[i][j]?"Y":"N")<< ")" << endl;
	}
	cout << "  ))" << endl;
	cout << ")" << endl;


}
