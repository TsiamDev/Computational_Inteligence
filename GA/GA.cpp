#include <iostream>
#include <stdio.h>
#include <cmath>
#include <iterator>
#include <algorithm>
#include <array>

#include <SDL2/SDL.h>

#define N 16//8
#define QN 16//8
#define POP_NUM 80//20
#define P2_BOUND 5
#define P1_BOUND (P2_BOUND - 2)


using namespace std;

std::array<std::array<int, N>, POP_NUM> pop = {-1};
std::array<int, POP_NUM> fitness = {-1};

std::array<std::array<int, N>, POP_NUM> new_pop = {-1};
std::array<int, POP_NUM> new_fitness = {-1};
int temp_indiv[N] = {-1};


int board[POP_NUM][N][N] = {0};
int queenPositions[POP_NUM][QN][2] = {-5};

//https://stackoverflow.com/questions/21890627/drawing-a-rectangle-with-sdl2
void PrintBoard(int pop_ind)
{
	//Initialize SDL and window
	int SDL_Init(SDL_INIT_EVERYTHING);
	SDL_Window* window = NULL;
	window = SDL_CreateWindow
	(
	"N-Queens", SDL_WINDOWPOS_UNDEFINED,
	SDL_WINDOWPOS_UNDEFINED,
	640,
	480,
	SDL_WINDOW_SHOWN
	);

	// Setup renderer
	SDL_Renderer* renderer = NULL;
	renderer =  SDL_CreateRenderer( window, -1, SDL_RENDERER_ACCELERATED);

	// Draw background
	// Set render color to brown ( background will be rendered in this color )
	SDL_SetRenderDrawColor( renderer, 101, 67, 33, 255 );
	// Clear winow
	SDL_RenderClear( renderer );

	//Draw the board -------------------------------------------------------------
	// Creat a rect at pos ( 50, 50 ) that's 50 pixels wide and 50 pixels high.
	SDL_Rect r;
	r.x = 0;
	r.y = 0;
	r.w = 50;
	r.h = 50;

	bool color = true;

	for(int i = 0; i < QN; i++)
	{	
		for(int j = 0; j < QN; j++)
		{
			if (color)
				SDL_SetRenderDrawColor( renderer, 255, 255, 255, 255 );	
			else
				SDL_SetRenderDrawColor( renderer, 0, 0, 0, 255 );

			color = not color;

			r.x += 50;
			// Render rect
			SDL_RenderFillRect( renderer, &r );
		}		
		r.y += 50;
		r.x = 0;
		if ( i % 2 == 1)
			color = true;
		else
			color = false;
	}

	//Draw the board (end) --------------------------------------------------------

	//Draw the queens----------------------

	r.w = 10;
	r.h = 10;
	SDL_SetRenderDrawColor( renderer, 0, 255, 0, 255 );
	//for (int i = 0; i < queenIndex-1; i++)
	for (int i = 0; i < QN; i++)
	{
		r.y = queenPositions[pop_ind][i][0] * 50 + 15 + r.w / 2;
		r.x = queenPositions[pop_ind][i][1] * 50 + 65 + r.h / 2;
		//cout << r.x << "," << r.y << endl;
		SDL_RenderFillRect( renderer, &r );
	}
	//Draw the queens (end) ---------------

	// Render the rect to the screen
	SDL_RenderPresent(renderer);

	//Check if [x] button is clicked and if so, exit
	//https://stackoverflow.com/questions/29706885/the-correct-way-to-test-sdl-pollevent-in-a-while
	SDL_Event e;
	int quit=0;
	while(!quit){
		while (SDL_PollEvent(&e)){
			if (e.type==SDL_QUIT)
				quit=1;
		}
	}

	SDL_DestroyWindow(window);
	SDL_Quit();

}

void Place(int x, int y, int pop_ind)
{
	if ( queenPositions[pop_ind][y][0] > -1 )
	{
		board[pop_ind][queenPositions[pop_ind][y][0]][queenPositions[pop_ind][y][1]] = 0;
	}

	//place on board
	//cout << "x " << x << "---" << "y " << y << endl;
	board[pop_ind][x][y] = 1;

	queenPositions[pop_ind][y][0] = x;
	queenPositions[pop_ind][y][1] = y;

}

void InitPop(void)
{
	for (int pi = 0; pi < POP_NUM; pi++)
	{
		for (int j = 0; j < N; j++)
		{
			pop[pi][j] = rand() % N;
			//cout << "pop[pi][j]: " << pop[pi][j] << endl;
			//cout << "j " << j << endl;
			//cout << "pi " << pi << endl;

			Place(pop[pi][j], j, pi);
		}
	}
}

bool CheckHorizontal(int x, int y, int pop_ind)
{
	bool flag = false;
	for (int i = 1; i < N; i++)
	{
		if( (y - i) < 0 )
		{
			//Out of bounds
			flag = false;
			break;
		}

		//cout << "board: " << board[pop_ind][x][y - i] << endl;

		if(board[pop_ind][x][y - i] == 1)
		{
			//has horizontal collision
		    	flag = true;
			return flag;			
		}
	}

	for (int i = 1; i < N; i++)
	{
		if( (y + i) >= N )
		{
			//Out of bounds
			flag = false;
			break;
		}

		//cout << "board[ " << pop_ind << "][" << x << "][" << y + i << "]: " << board[pop_ind][x][y + i] << endl;

		if(board[pop_ind][x][y + i] == 1)
		{
			//has horizontal collision
		    	flag =  true;
			return flag;
		}
	}

	return flag;
}

//Returns true if there is a vertical collision
//bool CheckVertical(int x, int y, int col)
bool CheckVertical(int x, int y, int pop_ind)
{
	//For given x position check all rows except the current
	for (int i = 0; i < N; i++)
	{
		//Exclude the current row
		if (i != x)
		{
			//cout << "board: " << board[pop_ind][i][y] << endl;
			if (board[pop_ind][i][y] == 1)
			{
				return true;
			}
		}
	}

	return false;
}

//returns true if there a collision exists
//otherwise returns false
bool CheckUpperLeftDiagonal(int x, int y, int pop_ind)
{
	for (int i = 1; i < N; i++)
	{
		if( ((x - i) < 0) or ((y - i) < 0) )
		{
			//Out of bounds
			return false;
		}

		//cout << "board: " << board[pop_ind][x - i][y - i] << endl;

		if(board[pop_ind][x - i][y - i] == 1)
		{
			//has collision at the upper left diagonal
		    	return true;
		}
	}

	return false;
}

bool CheckLowerRightDiagonal(int x, int y, int pop_ind)
{
	for (int i = 1; i < N; i++)
	{
		if (((x + i) >= N) or ((y + i) >= N))
		{
			//Out of bounds
			return false;
		}

		//cout << "board: " << board[pop_ind][x + i][y + i] << endl;

		if (board[pop_ind][x + i][y + i] == 1)
		{
			//has collision at lower right diagonal
			return true;
		}
	}

	return false;
}

bool CheckLowerLeftDiagonal(int x, int y, int pop_ind)
{
	for (int i = 1; i < N; i++)
	{
		if (((x + i) >= N) or ((y - i) < 0))
		{
			//Out of bounds
			return false;
		}

		//cout << "board: " << board[pop_ind][x + i][y - i] << endl;

		if (board[pop_ind][x + i][y - i] == 1)
		{
			//has collision at lower left diagonal
			return true;
		}
	}

	return false;
}

bool CheckUpperRightDiagonal(int x, int y, int pop_ind)
{
	for (int i = 1; i < N; i++)
	{
		if ( ((x - i) < 0) or ((y + i) >= N) )
		{
			//Out of bounds
			return false;
		}

		//cout << "board: " << board[pop_ind][x - i][y + i] << endl;

		if (board[pop_ind][x - i][y + i] == 1)
		{
			//has collision at upper right diagonal
			return true;
		}
	}

	return false;
}

//returns true if there is at least one collision
//otherwise returns false
bool CheckCollisions(int x, int y, int pop_ind)
{
	bool flag;
	//cout << "check horizontal" << endl;
	flag = CheckHorizontal(x, y, pop_ind);
	//cout << "check vertical" << endl;
	flag = flag | CheckVertical(x, y, pop_ind);
	//cout << "check upper left" << endl;
	flag = flag | CheckUpperLeftDiagonal(x, y, pop_ind);
	//cout << "check lower left" << endl;
	flag = flag | CheckLowerLeftDiagonal(x, y, pop_ind);
	//cout << "check upper right" << endl;
	flag = flag | CheckUpperRightDiagonal(x, y, pop_ind);
	//cout << "check lower right" << endl;
	flag = flag | CheckLowerRightDiagonal(x, y, pop_ind);
	//cout << "flag: " << flag << endl;
	//cout << "******************************" << endl;
	return flag;
}

int Getfitness(int x, int pop_ind, auto pop)
{
	int fit = 0;

	for(int y = 0; y < N; y++)
	{
		//cout << "checking " << x << "," << y << " for collisions" << endl;
		if( CheckCollisions(x, y, pop_ind) == false )
		{
			fit++;
		}
		//cout << "found " << 8 - fit << " collisions" << endl;
	}
		
	return fit;
}

void CalcFitness()
{
	for (int pi = 0; pi < POP_NUM; pi++)
	{
		int fit = 0;
		for(int x = 0; x < N; x++)
		{
			for(int y = 0; y < N; y++)
			{
				//cout << "checking " << x << "," << y << " for collisions" << endl;
				if( CheckCollisions(x, y, pi) == false )
				{
					fit++;
				}
				//cout << "found " << 8 - fit << " collisions" << endl;
			}
		}
		fitness[pi] = fit;
		//cout << "-------------------" << endl;
	}
	//return fit;
}

void CalcNewFitness()
{
	for (int pi = 0; pi < POP_NUM; pi++)
	{
		//cout << "board pi: " << pi << endl;
		int fit = 0;
		//cout << "fitness: " << fitness[pi] << endl;
		for(int x = 0; x < N; x++)
		{
			for(int y = 0; y < N; y++)
			{
				//cout << "checking " << x << "," << y << " for collisions" << endl;
				if( CheckCollisions(x, y, pi) == false )
				{
					fit++;
				}
				//cout << "found " << 8 - fit << " collisions" << endl;
			}
		}
		new_fitness[pi] = fit;
		//cout << "-------------------" << endl;
	}
	//return fit;
}

int HasConverged()
{
	for(int pi = 0; pi < POP_NUM; pi++)
	{
		if ( fitness[pi] == N)
			return pi;
	}	
	return -1;
}

int* Mate(int p1, int p2)
{
	int cross_point = rand() % N;
	temp_indiv[N] = {-1};
	//cout << "cross_point " << cross_point << endl;
	for(int i = 0; i < cross_point; i++)
	{
		temp_indiv[i] = pop[p1][i];
	}

	for(int i = cross_point; i < N; i++)
	{
		temp_indiv[i] = pop[p2][i];
	}

	return temp_indiv;
}

int* Mutate(int* new_indiv)
{
	for(int i = 0; i < N; i++)
	{
		int t = rand() % 100;
		double temp = (double)(t)/100;
		if ( temp < 0.05 )
		{
			int temp2 = rand() % 2;
			if ( temp2 == 0 )
			{
				new_indiv[i]++;
				if ( new_indiv[i] >= N )
					new_indiv[i] = N-1;
			}else{
				new_indiv[i]--;
				if ( new_indiv[i] < 0 )
					new_indiv[i] = 0;
			}
		}
	}
		
	return new_indiv;
}

int main(void)
{
	int seed = 42;
	srand(seed);
	//srand(time(NULL));

	cout << "Initialize population..." << endl;
	InitPop();

	cout << "Calculate Fitness..." << endl;
	CalcFitness();

	cout << "Start GA iterations..." << endl;
	//repeat untill convergence
	bool hasConvd = false;
	int index;
	int cnt = 0;
	int max_fitness = 0;
	do{
		//selection - crossover - mutation
		//we need POP_NUM new individuals for the new generation
		//#pragma omp parallel for
		for(int pi = 0; pi < POP_NUM; pi++)
		{
			//select two (different) parents stochastically
			//with "good" (relative to current max) fitness
			int p1;
			do{
				p1 = rand() % POP_NUM;
				if( fitness[p1] > (max_fitness - P1_BOUND) )
					break;
			}while( true );
			int p2;
			do{
				p2 = rand() % POP_NUM;
				if( fitness[p1] > (max_fitness - P2_BOUND) )
				{
					if ( p1 != p2 )
						break;
				}
			}while( true );

			//crossover
			int * new_indiv;
			new_indiv = Mate(p1, p2);
			//mutation
			new_indiv = Mutate(new_indiv);

			for(int j = 0; j < N; j++)
			{
				//insert new_pop
				new_pop[pi][j] = new_indiv[j];

				//update the board and the queenPositions
				Place(new_pop[pi][j], j, pi);
			}
		}
		//update pop
		std::copy(std::begin(new_pop), std::end(new_pop), std::begin(pop));

		//update fitness
		CalcNewFitness();
		std::copy(std::begin(new_fitness), std::end(new_fitness), std::begin(fitness));


		//convergence check
		index = HasConverged();
		if ( index != -1 )
		{
			hasConvd = true;
		}

		for (int pi = 0; pi < POP_NUM; pi++)
		{
			//cout << "fitness for pop " << pi << ": " << fitness[pi] << endl;
			if (fitness[pi] > max_fitness)
				max_fitness = fitness[pi];
		}
		//cout << "max fitness: " << max_fitness << endl;
		cnt++;
	}while( hasConvd == false);//cnt < 2000 );//hasConvd == false);//
	cout << "Finished." << endl;
	cout << "Pop " << index << " has converged" << endl;
	cout << "Took " << cnt << " iterations" << endl;

	cout << "Seed " << seed << endl;

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)	
			cout << board[index][i][j] << ' ';
		cout << endl;
	}
	cout << endl;
	
	//PrintBoard(index);
	return 0;
}