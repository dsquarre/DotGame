#include <stdio.h>
#include <stdlib.h>
#include <time.h>
struct Index{
	int a;
	int b;
};
struct Index makebox(int,int,int*);
void display(int,int*,char*);
int possiblemove(int,int*);
int gameover(int,int*,int,int);

int makestate(int,int*);
double evaluate(int);
int computermove(int,int*,int);



void main(){
	/*FILE *f;
	f = fopen("values.txt","w");
	fprintf(f,"%d\n%d\n%d",1,0,0);
	fclose(f);*/
	//take board size = 5
	printf("Enter dots in one side > ");
	int gridsize = 5;
	scanf("%d",&gridsize);
	int totalmoves = 2*(gridsize - 1)*gridsize;
	int grid[totalmoves];
	
	char boxes[(gridsize-1)*(gridsize-1)];
	
	for(int i = 0; i<totalmoves;i++){
		grid[i] = 1;
	}
	for(int i = 0; i<(gridsize-1)*(gridsize-1);i++){
		boxes[i] = ' ';
	}
	printf("\e[H\e[2J\e[3J"); //clear screen
	display(gridsize,grid,boxes);
	//game start
	int playerboxes = 0,computerboxes = 0;
	char turn = 'P'; int move;
	
	while(1){
		
		
		struct Index box;
		move = 0;
		if(turn == 'P'){
		turn = 'C';
		//player move
		while(1){
		printf("Enter your move >");
		scanf("%d",&move);
		if(grid[move]) break;
		printf("Already Filled Position");
		}
		//printf("\e[H\e[2J\e[3J"); //clear screen
		//move = computermove(totalmoves,grid,gridsize);
		grid[move] = 0;
		
		box = makebox(move,gridsize,grid);
		if(box.a != -1){ playerboxes +=1; boxes[box.a] = 'P'; turn = 'P';}
		if(box.b != -1){ playerboxes +=1; boxes[box.b] = 'P'; turn = 'P';}
		for(int i= 0;i<10000000;i++);//time control
		display(gridsize,grid,boxes);	
		if(gameover(totalmoves,grid,playerboxes,computerboxes)) break;
		}
		//printf("%d",box.b);
		if(turn == 'C'){
		turn = 'P';
		move = computermove(totalmoves, grid, gridsize);
		//printf("%d",move);
		grid[move] = 0;
		
		box = makebox(move,gridsize,grid);
		if(box.a != -1) {computerboxes +=1; boxes[box.a] = 'C'; turn = 'C';}
		if(box.b != -1) {computerboxes +=1; boxes[box.b] = 'C'; turn = 'C';}
		display(gridsize,grid,boxes);
		if(gameover(totalmoves,grid,playerboxes,computerboxes)) break;
		}
	}
	//end of program
	//this was for interpolation 
	/*
	int eval;
	if(playerboxes>computerboxes){eval = 1;}
	else if(playerboxes<computerboxes){eval = -1;}
	else{ eval = 0;}
	//since game is over and last move is stored
	int n;
	printf("storing the game sequence and value in values.txt");
	f = fopen("legacy/values.txt","r");
	fscanf(f,"%d",&n);
	int x[n]; int y[n];
	for(int i = 0;i<n;i++){
		eval = eval * move;
		fscanf(f,"%d",&x[i]);
		fscanf(f,"%d",&y[i]);
		}
	fclose(f);
	f = fopen("values.txt","w");
	fprintf(f,"%d\n",n+1);
	for(int i=0;i<n;i++){
		fprintf(f,"%d\n%d\n",x[i],y[i]);
	}
	fprintf(f,"%d\n%d",move,eval/move);
	fclose(f);
	//main();	*/
}

// move = rows(gridsize-1) + col(gridsize) + x
//assume row is horizontal, then row = col
//move = a*(2gridsize-1) +x thus a = move/(2g-1) and x = move%(2g-1)
//0<= x <= gridsize-2 for horizontal otherwise vertical
//so 

struct Index makebox(int move,int gridsize ,int grid[]){
	struct Index boxes;
	boxes.a = -1;
	boxes.b = -1; 
	int rows = (move)/(2*gridsize-1);
	int x = move%(2*gridsize-1);
	

	if( x >= 0 && x<= gridsize-2){	//move is horizontal !
		//row == col
		//possible boxes index = [(col-1) or (col) ]* (gridsize-1) + x;
		//for col-1 : check move, (row-1)(2*gridsize - 1) + x, move - gridsize, move - gridsize + 1
		//for col : check move, (row+1)(2*gridsize -1)+x, move + gridsize, move+gridsize -1
		if(rows !=0){ //for col - 1
			if(grid[move] || grid[(rows-1)*(2*gridsize-1)+x] || grid[move-gridsize] || grid[move-gridsize+1]) {}
			else {boxes.a = (rows-1)*(gridsize -1) + x;}
		}
		if(rows != gridsize -1){ //for col 
			if(grid[move] || grid[(rows+1)*(2*gridsize-1)+x] || grid[move+gridsize] || grid[move+gridsize-1]) {}
			else{
			if(boxes.a == -1) boxes.a = rows*(gridsize-1) + x;
			else boxes.b = rows*(gridsize-1) +x;
			}
		}	
	}
	
	else{   //move is vertical ~
		//rows = col+1;
		rows += 1;
		x = move - (rows*(gridsize-1) + (rows-1)*(gridsize));
		//possible boxes index = (row-1) *(gridsize -1) + (x or x-1)
		//for x-1 : check move, move-1, move - gridsize , move + gridsize - 1
		//for x : check move,move+1, move - (gridsize -1), move + gridsize
		
		if(x != 0){ //for x-1
			if(grid[move] ||   grid[move-1] || grid[move-gridsize] || grid[move+gridsize-1]){}
			else {boxes.a = (rows-1)*(gridsize-1)+x-1;}
		}
		
		if(x != gridsize){ //for x
			if(grid[move] || grid[move+1] || grid[move-gridsize+1] || grid[move+gridsize]) {}
			else{
				if(boxes.a == -1) boxes.a = (rows-1)*(gridsize-1)+x;
				else boxes.b = (rows-1)*(gridsize-1)+x; 
			}
		}
	}
	return boxes;
}


void display(int gridsize, int grid[], char boxes[]){
	int index = 0;
	char* draw = "\033[8m  \033[m";
	for(int i=0;i < gridsize*gridsize ;i++){
		printf("\033[42m  \033[m");
		if((i+1)%gridsize == 0){ 
			printf("\n"); 
			if(i == gridsize*gridsize -1) break;
			
			for(int j = 0;j<gridsize;j++){
				
				if(*(grid++) == 0) printf("\033[47m  \033[m");
				else printf("\033[33m%2d\033[m",index);
				index+=1;
				if(j<gridsize-1) { char c = *(boxes++); 
					if(c == 'P'){printf("\033[41m  \033[m");}
					else if(c == 'C'){printf("\033[44m  \033[m");}
					else{printf("\033[8m  \033[m");}}
			}
			printf("\n");
			continue;
		}
		
		if(*(grid++) == 0) printf("\033[47m  \033[m");
		else printf("\033[33m%2d\033[m",index);
		index+=1;
	}
	printf("\n\n");

}
int computermove(int totalmoves,int grid[],int gridsize){
	srand(time(NULL));
	for(int i=0;i<totalmoves;i++){
		if(grid[i]!=0){
			grid[i] = 0;
			struct Index box = makebox(i,gridsize,grid);
			grid[i] = 1;
			if(box.a !=-1 || box.b != -1){ return i;}
		}
	}
	int move = 0;
	int good = 1;
	int count = 0;
	while(1){
		move = rand()%(totalmoves-1);
		count+=1;
		good = 1;
		if(grid[move] !=0){
		    if(count<1000){
			grid[move] = 0;
			for(int i=0;i<totalmoves;i++){
				if(grid[i]!=0){
					grid[i] = 0;
					struct Index box = makebox(i,gridsize,grid);
					grid[i] = 1;
					if(box.a !=-1 || box.b != -1){ good = 0;}
				}
			}
			grid[move] = 1;
			if(good) {return move;}
		}
		else{return move;}
		   }
	}
	
	return 0;
}

int gameover(int totalmoves, int* grid, int playerboxes, int computerboxes){
	int over = 0;
	for(int i = 0; i<totalmoves; i++){
		if(*(grid++) != 0) return 0;
		over = 1;
	}
	if(over){
		if(playerboxes > computerboxes){
			printf("RED Won\n");
			}
		else if(playerboxes == computerboxes){
			printf("Game Draw\n");
			}
		else {
			printf("BLUE Won\n");
			}
		}
	return over;
}

//trying a function approximator using polynomial interpolation, didnt work sadly. Too frustrated to check
// same thing can be achieved faster and more efficiently using nn, so shifting to python Q Learning
int possiblemove(int totalmoves, int* grid){
	int i = 0;
	while(grid[i] == 0){
		i = rand()%(totalmoves-1) +1;
		}
	return i;
}


int makestate(int totalmoves,int* grid){
	int x = 0;
	for(int i = totalmoves - 1;i >= 0;i++){
		if (x > 999999999) return -1;
		
		if(grid[i]){
			x = x*100 + i; }
	}
	return x;	
}


double evaluate(int state){
	FILE *f; int n;
	f = fopen("values.txt","r");
	fscanf(f,"%d",&n);
	int x[n]; int y[n];
	for(int i = 0;i<n;i++){
		fscanf(f,"%d",&x[i]);
		fscanf(f,"%d",&y[i]);
		}
	fclose(f);
	
	double eval = 0.00;
	
		for(int i = 0; i<n; i++){
			double l = 1.00;
			for(int j = 0; j<n; j++){
				l *= (double)(state - x[j])/(double)(x[i]-x[j]);
			}
			eval += l*y[i]/(double)(state - x[i]);
		}
		for(int i = 0;i<n-1;i++){
			eval = eval/state;
		}
	
	return eval;
}


int bot(int totalmoves, int* grid, int myboxes, int oppboxes){
	double eval = myboxes - oppboxes + evaluate(makestate(totalmoves,grid));
	int tempgrid[totalmoves];
	for(int i = 0; i<totalmoves; i++){
		tempgrid[i] = grid[i]; 
	}
	
	int move = possiblemove(totalmoves,tempgrid);
	grid[move] = 0;
	int best = move;
	double besteval = myboxes - oppboxes + evaluate(makestate(totalmoves,grid));
	grid[move] = 1;
	while(1){
		if(besteval>eval){best = move;}
		move = possiblemove(totalmoves,tempgrid);
		if(move = -1) break;
		grid[move] = 0;
		eval = myboxes - oppboxes + evaluate(makestate(totalmoves,grid));
		grid[move] = 1;
	}
	return best;
}

