import pickle
import os.path
import math
import matplotlib.pyplot as plt

class recommender:
  def __init__(self):
    self.trainData = pickle.load( open( "trainData.p", "rb" ) )
    self.testData  = pickle.load( open( "testData.p", "rb" ) )
    if os.path.isfile('slope_one_diffs.p') and os.path.isfile('slope_one_freqs.p'): 
      self.slope_one_diffs = pickle.load( open( "slope_one_diffs.p", "rb" ) )
      self.slope_one_freqs = pickle.load( open( "slope_one_freqs.p", "rb" ) )
    else:
      self.slope_one_diffs = {}
      self.slope_one_freqs = {}
      self.slope_one_update(self.trainData)
  
  def slope_one_predict(self, userdata):
    preds, freqs, result = {}, {}, {}
    for key_user,userprefs in userdata.iteritems():
      for item, rating in userprefs.iteritems():
          for diffitem, diffratings in self.slope_one_diffs.iteritems():
              try:
                  freq = self.slope_one_freqs[diffitem][item]
              except KeyError:
                  continue
              preds.setdefault(diffitem, 0.0)
              freqs.setdefault(diffitem, 0)
              preds[diffitem] += freq * (diffratings[item] + rating)
              freqs[diffitem] += freq
      result[key_user]=dict([(item, value / freqs[item])
                             for item, value in preds.iteritems()
                             if item not in userprefs and freqs[item] > 0])
    return result
  
  def slope_one_update(self, userdata):
    for ratings in userdata.itervalues():
        for item1, rating1 in ratings.iteritems():
            self.slope_one_freqs.setdefault(item1, {})
            self.slope_one_diffs.setdefault(item1, {})
            for item2, rating2 in ratings.iteritems():
                self.slope_one_freqs[item1].setdefault(item2, 0)
                self.slope_one_diffs[item1].setdefault(item2, 0.0)
                self.slope_one_freqs[item1][item2] += 1
                self.slope_one_diffs[item1][item2] += rating1 - rating2
    for item1, ratings in self.slope_one_diffs.iteritems():
        for item2 in ratings:
            ratings[item2] /= self.slope_one_freqs[item1][item2]
    pickle.dump( self.slope_one_diffs, open( "slope_one_diffs.p", "wb" ) )
    pickle.dump( self.slope_one_freqs, open( "slope_one_freqs.p", "wb" ) )
  
  def slope_one_error(self,method='RMSE'):
    if method is 'RMSE':
      result = self.slope_one_predict(self.trainData)
      error  = {}
      count  = {}
      total_count  = 0
      total_error  = 0.0
      for key_user,pref_user in self.testData.iteritems():
        error[key_user] = 0.0
        count[key_user] = 0
        for key_user1,ans_user in pref_user.iteritems():
          try:
            error[key_user] += ((result[key_user][key_user1]-ans_user)**2)
            count[key_user] += 1
          except KeyError:
            continue
        total_count += count[key_user]
        total_error += error[key_user]
        error[key_user] = math.sqrt(error[key_user]/count[key_user])
      total_error = math.sqrt(total_error/total_count)
      plt.figure()
      x = error.keys()
      y = error.values()
      plt.plot(x,y,'ro')
      plt.title('Error plot')
      plt.show()
      plt.figure()
      x = count.keys()
      y = count.values()
      plt.plot(x,y,'ro')
      plt.title('Count plot')
      plt.show()
    return total_error
#if __name__ == '__main__':
#  s = recommander()
