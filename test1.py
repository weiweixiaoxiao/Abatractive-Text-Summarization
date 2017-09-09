import os
def readTXT(inFile):
    reader = open(inFile, 'r',encoding = 'UTF-8')
    contentlines = reader.readlines()
    j = 0
    for str1 in contentlines:
        strfile = "articleContent/article"+str(j)+"/"+str(j)+".txt"
        strfiledir = "E://article"+str(j)
        os.mkdir(strfiledir)
        outFile = strfile
        print("j = {} outFile = {} ".format(j,outFile))
        j += 1
        outWriter = open(outFile, 'w',encoding = 'UTF-8')
        strlist = str1.split()
        string = ''
        for i in range(len(strlist)): 
            if strlist[i] == "。" or strlist[i] == ";" or strlist[i] == "!":    
                outWriter.write(string)    
                outWriter.write('\n')
                string = ''           
            else:
                string += strlist[i]
        outWriter.write(string) 
        outWriter.close()    
    reader.close()
    

readTXT('dataTest/articleTrainData.txt')   
print("end")