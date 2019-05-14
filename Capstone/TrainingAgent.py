
# coding: utf-8

# In[13]:



#We had our first two heuristics (Misplaced Tile and TileRowCol) to be multiples of 5 to our implementation of the
#Neural Network so we multpied the values by a given weight of 5. This is because we didnt want the neural network
#to be skewered to the Nilson Heuristic.
#Futre work may decrease this to 1 to have no effect as A* and the Neural Network heuristic both use the heuristics
weight=5

    
    
#This heuristic is the number of tiles that are misplaced from their correct positions.    
def MisplacedTile(state,size):
    count=0
    for i, c in enumerate(state):
        if(c!="_"):
            tile=int(c, 16) 
            if ((i+1)!=(tile)):
                count+=1
    if(size==11):
        if(state[11]=="_"):
            count-=1
    if(size==15):
        if(state[15]=="_"):
            count-=1
    return count*weight

#This heuristic is the sum of number of tiles not in their correct rows
#and number of tiles not in their correct columns. 
def TileRowCol(state,size):
    row=0
    col=0
    if(size==11):
    #Check Row then Col
 
        for i, c in enumerate(state):
             if(c!="_"):
                tile=int(c, 16) 
                if(tile<5):
                    if (not((i+1)<5)):
                        row+=1
                   
                if((tile in range(5,9))):
                    if not((i+1) in range(5,9)):
                        row+=1 
                if((tile in range(9,12))):
                    if not((i+1) in range(9,12)):
                        row+=1 
                if not(((i+1)== tile) or ((i+1)== tile+4) or ((i+1)== tile+8)):
                    col+=1
                    
             else:
                if not((i+1) in range(9,12)):
                    row+=1 
                if not(((i+1)== 4) or ((i+1)== 8) or ((i+1)== 12)):
                    col+=1
                
        return (col+row)*weight
    elif(size==15):
    #Check Row then Col
        for i, c in enumerate(state):
             if(c!="_"):
                tile=int(c, 16) 
                if(tile<5):
                    if (not((i+1)<5)):
                        row+=1   
                if((tile in range(5,9))):
                    if not((i+1) in range(5,9)):
                        row+=1 
                if((tile in range(9,12))):
                    if not((i+1) in range(9,12)):
                        row+=1 
                if((tile in range(12,16))):
                    if not((i+1) in range(12,16)):
                        row+=1 
                if not(((i+1)== tile) or ((i+1)== tile+4) or ((i+1)== tile+8) or ((i+1)== tile+12)):
                    col+=1
                    
             else:
                if not((i+1) in range(12,16)):
                    row+=1 
                if not(((i+1)== 4) or ((i+1)== 8) or ((i+1)== 12) or (i+1)==16):
                    col+=1
                
        return (col+row)*weight              
#
#This is the sum of the horizontal and vertical distances of each tile from their goal positions.
def ManDist(state,size):
    finalsum=0
    for i, c in enumerate(state):
        if (c!="_"):
            value=VertD(i,c,state,size)+HorD(i,c,state,size)
            finalsum+=value
    return finalsum

#The horizontal distance of a tile to its goal position. 
def HorD(index,tile,state,size):
    if (size==11):
        tilepos=int(tile, 16)
        if (tilepos>8):
            tilepos-=8
        if(tilepos>4):
            tilepos-=4
        index+=1   
        if (index>8):
            index-=8
        if(index>4):
            index-=4
        
        return abs(tilepos-index)
    if (size==15):
        tilepos=int(tile, 16)
        if (tilepos>12):
            tilepos-=12
        if (tilepos>8):
            tilepos-=8
        if(tilepos>4):
            tilepos-=4
        index+=1  
        if (index>12):
            index-=12
        if (index>8):
            index-=8
        if(index>4):
            index-=4
        
        return abs(tilepos-index)   
        
#The vertical distance of a tile to its goal position.         
def VertD(index,tile,state,size):
    ypos=0
    curpos=0
    if (size==11):
        tile=int(tile, 16)
        if(tile>8):
            ypos=3
        elif(tile>4):
            ypos=2
        else:
            ypos=1
        index+=1
        if(index>8):
            curpos=3
        elif (index>4):
            curpos=2
        else:
            curpos=1
       
        return abs(ypos-curpos)
        
    elif (size==15):
        tile=int(tile, 16)
        if(tile>12):
            ypos=4
        if(tile>8):
            ypos=3
        elif(tile>4):
            ypos=2
        else:
            ypos=1
        index+=1
        if(index>12):
            curpos=4
        if(index>8):
            curpos=3
        elif (index>4):
            curpos=2
        else:
            curpos=1
       
        return abs(ypos-curpos)
        
 
 #This heuristic adds a Sequence Score to the Manhattan Distance heuristic.
# The score is made of  a cost of 2 for every non blank tile  not followed by its proper
#successor tile and a cost of 1 if the blank is not in it's proper position 
def Nilson(state,size):
    score=0
    for i, c in enumerate(state):
        if (c!="_"):
            tile=int(c, 16) 
            if ((i+1)!=tile):
                score+=1
        if(c=="_"):
                if not((i==11) or (i==15)):
                    score+=1
    return ManDist(state,size) + (3*score)
                    
#This heuristic adds a cost of 2 to the Manhattan Distance for every pair of conflicting tiles in the board configuration. 
#Two tiles j and k are in a linear conflict if j and k are in the same line, the goal positions 
#of both tiles are also in that line, 
#j is to the right of k and goal position of j is to the left of the goal position of k.            
def LinearCon(state,size):
    start=0
    stop=4
    cost=0
    while(start!=size+1):
        pairs= [(state[k], state[j]) for k in range(start,stop) for j in range(k+1,stop)]
        delist=[]
        for i in pairs:
            if ("_" in i):
                delist.append(i)
        for i in delist:        
            pairs.remove(i)
        for pair in pairs:
            tilek,tilej=pair
            index1=state.index(tilek)
            index2=state.index(tilej)
            cost+=ConCk(tilek,index1,tilej,index2,state,size)
        start+=4
        stop+=4
    return ManDist(state,size) + cost
            
               
                    
                
#Checks if there is a linear conflict between the two tiles in a puzzle state                
def ConCk(tile1,index1,tile2,index2,state,size):
    if((VertD(index1,tile1,state,size)==0) and ((VertD(index2,tile2,state,size)==0))):
        tile1k=int(tile1, 16)
        tile2j=int(tile2,16)
        if(tile2j<tile1k):
            return 2
        else:
            return 0
    else:
        return 0
        
 #Needed to get the legal moves for a puzzle state
def pMoves(state,blank,size):
    moves=[]
    if(size==11):
        #if blank can move right then get the tuple for that move
        #maybe change to even number with division
        if(((blank+1)!=4) and ((blank+1)!=8) and ((blank+1)!=12)):
            #moves.append((blank,state[blank+1]))
            moves.append(1)
        #if the blank can move left get the tuple for that move
        if(((blank-1)!=-1) and ((blank-1)!=3)and((blank-1)!=7)):
            #moves.append((blank,state[blank-1])) 
            moves.append(-1)
        #if the blank can move up then get the tuple for that move
        if((blank+1)>=4):
            #moves.append((blank,state[blank-4]))
            moves.append(-4)
        #if the blank can move down then get tuple for that move
        if((blank+1)<=8):
           #moves.append((blank,state[blank+4]))
           moves.append(4)
        return moves
    if(size==15):
        #if blank can move right then get the tuple for that move
        #maybe change to even number with division
        if(((blank+1)!=4) and ((blank+1)!=8) and ((blank+1)!=12) and ((blank+1)!=16)):
            #moves.append((blank,state[blank+1]))
            moves.append(1)
        #if the blank can move left get the tuple for that move
        if(((blank-1)!=-1) and ((blank-1)!=3)and((blank-1)!=7) and((blank-1)!=11)):
            #moves.append((blank,state[blank-1])) 
            moves.append(-1)
        #if the blank can move up then get the tuple for that move
        if((blank+1)>=4):
            #moves.append((blank,state[blank-4]))
            moves.append(-4)
        #if the blank can move down then get tuple for that move
        if((blank+1)<=12):
           #moves.append((blank,state[blank+4]))
           moves.append(4)
        return moves
    
#This function returns the output move for each puzzle state if 
#the A* algorithm with the admissible Linear Conflict heuristic
#expands a node with this state
def AdHeurOutput(state,size):
    blank=state.index("_")
    pStates=pMoves(state,blank,size)
    bestH=5000
    bestmove=0
    test=[]
    for move in pStates:
        newstate=list(state)
        newstate[blank],newstate[blank+move]=newstate[blank+move],newstate[blank]
        curH=LinearCon(newstate,size)
        test.append((state,curH,bestH,move))
        if (curH<bestH):
            bestH=curH
            bestmove=move
    if (bestmove==1):
        bestmove=3
    if (bestmove==(-1)):
        bestmove=1
    if (bestmove==4):
        bestmove=0
    if (bestmove==(-4)):
        bestmove=2
    return bestmove
        
                

#Creates a training dataset file and corresponding labels file for the neural network.
def Heuristics(file,size):
    if(size==11):
        input=open(file,"r")
        output=open("11DataSet.txt","w")
        label=open("11DataLabel.txt","w")
    if(size==15):
        input=open(file,"r")
        output=open("15DataSet.txt","w")
        label=open("15DataLabel.txt","w")
    for state in input:
        state=state.strip('\n')
        size=len(state)-1
        sub1=MisplacedTile(state,size)
        sub2=TileRowCol(state,size)
        sub3=Nilson(state,size)   
        adh=AdHeurOutput(state,size)
        output.write(str([sub1,sub2,sub3])+"\n")
        label.write(str([adh])+"\n")

    

    
#Run this function to create a example dataset and label files
#Heuristics("11puzzles.txt",11)

