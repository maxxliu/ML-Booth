
##################### Getting the Data#######################

PackageList =c('MASS','gbm','tree','randomForest','rpart','caret','ROCR','readxl','data.table','R.utils') 
NewPackages=PackageList[!(PackageList %in% 
                            installed.packages()[,"Package"])]
if(length(NewPackages)) install.packages(NewPackages)
lapply(PackageList,require,character.only=TRUE)#array function



gitURL="https://github.com/ChicagoBoothML/MLClassData/raw/master/KDDCup2009_Customer_relationship/";
DownloadFileList=c("orange_small_train.data.gz","orange_small_train_appetency.labels.txt",
                   "orange_small_train_churn.labels.txt","orange_small_train_upselling.labels.txt")
LoadFileList=c("orange_small_train.data","orange_small_train_appetency.labels.txt",
               "orange_small_train_churn.labels.txt","orange_small_train_upselling.labels.txt")

for (i in 1:length(LoadFileList)){
  if (!file.exists(LoadFileList[[i]])){
    if (LoadFileList[[i]]!=DownloadFileList[[i]]) {
      download.file(paste(gitURL,DownloadFileList[[i]],sep=""),destfile=DownloadFileList[[i]])
      gunzip(DownloadFileList[[i]])
    }else{
      download.file(paste(gitURL,DownloadFileList[[i]],sep=""),destfile=DownloadFileList[[i]])}}
}

na_strings <- c('',
                'na', 'n.a', 'n.a.',
                'nan', 'n.a.n', 'n.a.n.',
                'NA', 'N.A', 'N.A.',
                'NaN', 'N.a.N', 'N.a.N.',
                'NAN', 'N.A.N', 'N.A.N.',
                'nil', 'Nil', 'NIL',
                'null', 'Null', 'NULL')

X=as.data.table(read.table('orange_small_train.data',header=TRUE,
                           sep='\t', stringsAsFactors=TRUE, na.strings=na_strings))
Y_churn    =read.table("orange_small_train_churn.labels.txt", quote="\"")
Y_appetency=read.table("orange_small_train_appetency.labels.txt", quote="\"")
Y_upselling=read.table("orange_small_train_upselling.labels.txt", quote="\"")










##################### All codes below are sample hints#######################
#1. How to write a loop over columns of a data.frame (say, X here)?

for (i in names(X)){
  CurrentColumn=X[[i]]
  CurrentColumnVariableName=i
  #Then you do the computation on CurrentColumn, using function is.na, and save the result
  cat(i, mean(is.na(CurrentColumn)),'\n')
}



#2. How to drop columns of a data.frame indexed by a list?

ExcludeVars=c('Var1','Var2','Var100') #for example
idx=!(names(X) %in% ExcludeVars);
XS=X[,!(names(X) %in% ExcludeVars),with=FALSE]



#3. How to convert missing values into factors?

i="Var208" #for example

CurrentColumn=XS[[i]]                    #Extraction of column
idx=is.na(CurrentColumn)                 #Locate the NAs
CurrentColumn=as.character(CurrentColumn)#Convert from factor to characters
CurrentColumn[idx]=paste(i,'_NA',sep="") #Add the new NA level strings
CurrentColumn=as.factor(CurrentColumn)   #Convert back to factors
XS[[i]]=CurrentColumn                    #Plug-back to the data.frame



#4. How to aggregate a number of factors into new factors?
Thres_Low=249;
Thres_Medium=499;
Thres_High=999;
i="Var220" #for example, this one has 4291 levels
CurrentColumn=XS[[i]]                    #Extraction of column

CurrentColumn_Table=table(CurrentColumn) #Tabulate the frequency
levels(CurrentColumn)[CurrentColumn_Table<=Thres_Low]=paste(i,'_Low',sep="")

CurrentColumn_Table=table(CurrentColumn) #Tabulate the new frequency 
levels(CurrentColumn)[CurrentColumn_Table>Thres_Low & CurrentColumn_Table<=Thres_Medium ]=paste(i,'_Medium',sep="")

CurrentColumn_Table=table(CurrentColumn) #Tabulate the new frequency
levels(CurrentColumn)[CurrentColumn_Table>Thres_Medium & CurrentColumn_Table<=Thres_High ]=paste(i,'_High',sep="")

XS[[i]]=CurrentColumn                    #Plug-back to the data.frame
#We got 9 levels after cleaning