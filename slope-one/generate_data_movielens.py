import pickle
import codecs

class generate_data:
    def MovieLens(self,path=''):
      trainData = {}
      testData  = {}
      f = codecs.open('../ml-100k/u1.base','r','ascii')
      for line in f:
        fields = line.split('\t')
        user = fields[0]
        movie = fields[1]
        rating = int(fields[2].strip().strip('"'))
        if user in trainData:
          curr = trainData[user]
        else:
          curr = {}
        curr[movie] = rating
        trainData[user]  = curr
      f.close()
      f = codecs.open('../ml-100k/u1.test','r','ascii')
      for line in f:
        fields = line.split('\t')
        user = fields[0]
        movie = fields[1]
        rating = int(fields[2].strip().strip('"'))
        if user in testData:
          curr = testData[user]
        else:
          curr = {}
        curr[movie] = rating
        testData[user]  = curr
      f.close()
      f = open('trainData.p','wb')
      pickle.dump(trainData,f)
      f.close()
      f = open('testData.p','wb')
      pickle.dump(testData,f)
      f.close()

if __name__ == '__main__':
    s = generate_data()
    s.MovieLens()
