import tensorflow as tf;
import numpy as np;
from vcdparser import VcdParser;

class TrainningFramework:
  def __init__(self, vcdfile):
    self.vcd = VcdParser(vcdfile);
  def selectInstance(self, instName, inputs = []):
    lenInstName = len(instName);
    self.symbols = [];
    self.inputs  = [];
    self.notinputs = [];

    self.symrefs = [];
    self.inputrefs = [];
    self.notinputrefs = [];
    for symbol in self.vcd.symbols:
      if (len(symbol) > lenInstName and symbol[:lenInstName] == instName):
        self.symbols.append(symbol);
        ref = self.vcd.symbols[symbol]['ref'];
        self.symrefs.append(ref);
    for input in inputs:
      inputSymbol = instName + '.' + input;
      if (inputSymbol in self.symbols):
        self.inputs.append(inputSymbol);
        ref = self.vcd.symbols[inputSymbol]['ref'];
        self.inputrefs.append(ref);
    for symbol in self.symbols:
      if (symbol not in self.inputs):
        self.notinputs.append(symbol);
        ref = self.vcd.symbols[symbol]['ref'];
        self.notinputrefs.append(ref);
  def _lenVcdValue(self, symbols):
    sz = 0;
    for symbol in symbols:
      sz += self.vcd.symbols[symbol]['length'];
    return sz;
  def lenData(self):
    return self._lenVcdValue(self.symbols);
  def lenInputs(self):
    return self._lenVcdValue(self.inputs);
  def lenNotInputs(self):
    return self._lenVcdValue(self.notinputs);
  def reset(self):
    self.vcd.restart();
  def window(self, N, clk, rsts):
    succ = self.vcd.parseWindow(N, clk, rsts);
    return succ;
  def _getVcdValue(self,N,refs):
    value = "";
    for ref in refs:
      value += self.vcd.windowValues[N][ref];
    return value;
  def getDataVcdValue(self,N):
    return self._getVcdValue(N,self.symrefs);
  def getInputVcdValue(self,N):
    return self._getVcdValue(N,self.inputrefs);
  def getNotInputVcdValue(self,N):
    return self._getVcdValue(N,self.notinputrefs);

class ModelFramework:
  def __init__(self, knownCycles = [], expectCycles = [], learnCycles = []):
    self.windowSize = max(knownCycles + expectCycles + learnCycles) + 1;
    for cycle in knownCycles:
      assert(cycle not in learnCycles);
      assert(cycle not in expectCycles);
    self.knownCycles  = knownCycles;
    self.learnCycles  = learnCycles;
    self.expectCycles = expectCycles;
    self.train_opt    = None;
  def setTrainningVcd(self, vcdfile, inst, inputs):
    self.trace = TrainningFramework(vcdfile);
    self.trace.selectInstance(inst, inputs);
    self.X_len = self.trace.lenData()      * len(self.knownCycles) + \
                 self.trace.lenNotInputs() * len(self.expectCycles);
    self.Y_len = self.trace.lenInputs()    * len(self.learnCycles);
  def placeholders(self):
    self.X      = tf.placeholder(dtype=tf.float32, shape=[1, self.X_len], name='X');
    self.Y_ref  = tf.placeholder(dtype=tf.float32, shape=[1, self.Y_len], name='Y');
    self.Y_mask = tf.placeholder(dtype=tf.float32, shape=[1, self.Y_len], name='Y');
  def maskstr(self, str):
    v = [];
    for s in str:
      if (s == '0' or s == '1'):
        v.append(1);
      if (s == 'x' or s == 'z'):
        v.append(0);
    return v;
  def convertstr(self, str):
    v = [];
    for s in str:
      if (s == '0' or s == 'z'):
        v.append(0);
      if (s == '1' or s == 'x'):
        v.append(1);
    return v;
  def getTraceX(self):
    xstr = '';
    for cycle in self.knownCycles:
      xstr += self.trace.getDataVcdValue(cycle);
    for cycle in self.expectCycles:
      xstr += self.trace.getNotInputVcdValue(cycle);
    mask = self.maskstr(xstr);
    rand = [];
    import random;
    for m in mask:
      rand.append(random.random() if m == 0 else 0);
    init = self.convertstr(xstr);
    return [np.add(init, rand)];
  def getTraceY(self):
    ystr = '';
    for cycle in self.learnCycles:
      ystr += self.trace.getInputVcdValue(cycle);
    return [self.convertstr(ystr)];
  def getMaskY(self):
    ystr = '';
    for cycle in self.learnCycles:
      ystr += self.trace.getInputVcdValue(cycle);
    return [self.maskstr(ystr)];
  def init(self):
    self.session = tf.Session();
    init = tf.initialize_all_variables();
    self.session.run(init);
  def train(self, clk, rsts):
    if (self.train_opt):
      loss = 0;
      while (self.trace.window(self.windowSize, clk, rsts)):
        placehoders = {
          self.X:     self.getTraceX(),
          self.Y_ref: self.getTraceY(),
          self.Y_mask:self.getMaskY()
        };
        self.session.run(self.train_opt, placehoders);
        loss += self.session.run(self.loss, placehoders);
      return loss;
    return None;

if __name__ == "__main__":
  class LinearModel (ModelFramework):
    def __init__(self, knownCycles, expectCycles, learnCycles):
      ModelFramework.__init__(self,knownCycles,expectCycles,learnCycles);
    def createVariable(self):
      self.W = tf.Variable(tf.random_uniform([self.X_len, self.Y_len], -1.0, 1.0));
      self.B = tf.Variable(tf.zeros([1, self.Y_len]));
    def createModel(self):
      self.Y = tf.matmul(self.X, self.W) + self.B;
    def createLoss(self):
      self.Y_diff = self.Y - self.Y_ref;
      self.Y_validDiff = self.Y_mask * self.Y_diff;
      self.loss = tf.reduce_mean(tf.square(self.Y_validDiff));
    def createOptimizer(self):
      self.optimizer = tf.train.GradientDescentOptimizer(0.04);
      self.train_opt = self.optimizer.minimize(self.loss);
    def create(self):
      self.placeholders();
      self.createVariable();
      self.createModel();
      self.createLoss();
      self.createOptimizer();
  train = TrainningFramework('./case/arbiter.vcd');
  train.selectInstance('top.U', ['clk','rst','req0','req1','req2','req3']);
  print ("Symbols");
  for sym in zip(train.symbols, train.symrefs):
    print ("\t" + sym[0] + "\t" + sym[1]);
  print ("Inputs");
  for sym in zip(train.inputs, train.inputrefs):
    print ("\t" + sym[0] + "\t" + sym[1]);
  while (train.window(3, {'symbol':'top.U.clk','edge':'posedge'}, [{'symbol':'top.U.rst','edge':'high'}])):
    print(train.lenInputs()),
    print(train.lenNotInputs()),
    print(train.getInputVcdValue(0)),
    print(train.getNotInputVcdValue(0)),
    print(train.getInputVcdValue(1)),
    print(train.getNotInputVcdValue(1)),
    print(train.getInputVcdValue(2)),
    print(train.getNotInputVcdValue(2));
  model = LinearModel([0,1,2],[4],[3]);
  model.setTrainningVcd('./case/arbiter.vcd', 'top.U', ['clk','rst','req0','req1','req2','req3']);
  print(model.X_len),
  print(model.Y_len);

  model.create();
  model.init();
  count = 2000;
  while (count):
    loss = model.train({'symbol':'top.U.clk','edge':'posedge'}, [{'symbol':'top.U.rst','edge':'high'}]);
    model.trace.reset();
    print (loss);
    if (loss < 3e-13):
      break;
    count = count - 1;

