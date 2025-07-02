class Env:
    dots = 0
    grid = [0]
    boxes = [0]
    def clear_screen(self):
        import os
        os.system('cls' if os.name == 'nt' else 'clear')

    def __init__(self,dots):
        self.dots = dots
        self.boxes = [0]*(dots-1)*(dots-1)
        self.grid = [0]*2 * dots *(dots - 1)

    def step(self,action,turn):
        self.grid[action] = 1
        rows,x = 0,0
        reward = 0.0
        rows = (action)//(2*self.dots-1)
        x = (action)%(2*self.dots-1)
        #print(rows,x,action,turn)
        if(x >= 0 and x<= self.dots-2):	#move is horizontal !
            # no. of  row == col
            #possible boxes index = [(col-1) or (col) ]* (gridsize-1) + x;
            #for col-1 : check move, (row-1)(2*gridsize - 1) + x, move - gridsize, move - gridsize + 1
            #for col : check move, (row+1)(2*gridsize -1)+x, move + gridsize, move+gridsize -1
            if(rows !=0): #for col - 1
                if(not (self.grid[action]==0 or self.grid[(rows-1)*(2*self.dots-1)+x]==0 or self.grid[action-self.dots]==0 or self.grid[action-self.dots+1]==0)):
                    index = (rows-1)*(self.dots -1) + x
                    #print(index)
                    self.boxes[index] = turn
                    reward = turn
            if(rows != self.dots -1): # for col
                if(not(self.grid[action]==0 or self.grid[(rows+1)*(2*self.dots-1)+x]==0 or self.grid[action+self.dots]==0 or self.grid[action+self.dots-1]==0)):
                    #print(rows)
                    index = rows*(self.dots-1) +x
                    self.boxes[index] = turn
                    reward = turn
        else:   #move is vertical ~
		#rows = col+1
            rows += 1
            x = action - (rows*(self.dots-1) + (rows-1)*(self.dots))
            #possible boxes index = (row-1) *(gridsize -1) + (x or x-1)
            #for x-1 : check move, move-1, move - gridsize , move + gridsize - 1
            #for x : check move,move+1, move - (gridsize -1), move + gridsize

            if(x != 0): #for x-1
                if(not(self.grid[action]==0 or self.grid[action-1]==0 or self.grid[action-(self.dots)]==0 or self.grid[action+self.dots-1]==0)):
                    index = (rows-1)*(self.dots-1)+x-1
                    self.boxes[index] = turn
                    reward = turn
            if(x != self.dots -1): #for x
                #print(x)
                if(not(self.grid[action]==0 or self.grid[action+1]==0 or self.grid[action-self.dots+1]==0 or self.grid[action+self.dots]==0)):
                    index = (rows-1)*(self.dots-1)+x
                    self.boxes[index] = turn
                    reward = turn

        return reward

    def render(self):
        #self.clear_screen() #clear screen
        boxindex = 0
        gridindex = 0

        for i in range(self.dots*self.dots):
            print("\033[42m  \033[m",end='')
            if((i+1)%self.dots == 0):
                print('\n',end='')
                if(i == self.dots*self.dots -1):
                    break
                for j in range(self.dots):
                    if(self.grid[gridindex] == 1):
                        print("\033[47m  \033[m",end='')
                    else:
                        print(f"\033[40m{gridindex:2d}\033[m",end='')
                    gridindex += 1
                    #print(gridindex,end='')
                    if(j<self.dots-1):
                        #print(boxindex)
                        box = self.boxes[boxindex]
                        boxindex += 1
                        if(box == 1):
                            print("\033[41m  \033[m",end='')
                        elif(box == -1):
                            print("\033[44m  \033[m",end='')
                        else:
                            print("\033[100m  \033[m",end='')
                print("\n",end='')
                continue
            if(self.grid[gridindex] == 1):
                print("\033[47m  \033[m",end='')
            else:
                print(f"\033[40m{gridindex:2d}\033[m",end='')
            gridindex += 1
        print('\n')

    def gameover(self):
        for i in range(len(self.grid)):
            if(self.grid[i]==0):
                return False
        return True

    def reset(self):
        self.grid = [0]*(2*self.dots*(self.dots-1))
        self.boxes = [0]*(self.dots-1)*(self.dots-1)
        return self.grid.copy()

    def action_space(self):
      actions = []
      for i in range(len(self.grid)):
        if(self.grid[i] == 0):
          actions.append(i)
      return actions
    
    @classmethod
    def from_state(cls, state):
        env = cls(dots=4) 
        g_len = len(env.grid)
        b_len = len(env.boxes)
        env.grid = list(state[:g_len])
        env.boxes = list(state[g_len:g_len + b_len])
        env.turn = state[-1]
        return env
    @classmethod
    def Gameover(cls, state):
        env = cls(dots=4) 
        g_len = len(env.grid)
        b_len = len(env.boxes)
        env.grid = list(state[:g_len])
        env.boxes = list(state[g_len:g_len + b_len])
        env.turn = state[-1]
        return env.gameover()
    
    def clone(self):
        env = Env(self.dots)
        for i in range(len(self.grid)):
            env.grid[i] = self.grid[i]
            if (i < len(self.boxes)):
                env.boxes[i] = self.boxes[i]
        return env
    
#testing environment
'''
obj = Env(4)
turn = -1
while(not obj.gameover()):
  if turn ==-1:
    move = input("enter move")
    reward = obj.step(int(move),turn)
    obj.render()
  elif turn  == 1:
    move = obj.minmax(turn)
    reward = obj.step(move,turn)
    obj.render()
  if reward == 0:
    turn = -turn
'''