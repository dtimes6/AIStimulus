import re;

class VcdParser:
  def __init__(self, filename):
    self.filename = filename;
    self.filehandler = open(self.filename);
    if (self.filehandler):
      self.lines = self.filehandler.readlines();
      self.parseVcdHeader();
      self.restart();
  def parseVcdHeader(self):
    self.linenumber = 0;
    self.symbols = {};
    self.refs = {};
    scope = [];
    for line in self.lines:
      self.linenumber += 1;
      if (line[0] == "$"):
        if (line[:6] == "$scope"):
          m = re.split(r'\s+', line);
          scope.append(m[2]);
        if (line[:8] == "$upscope"):
          del scope[:-1];
        if (line[:4] == "$var"):
          m = re.split(r'\s+', line);
          type   = m[1];
          length = int(m[2]);
          ref    = m[3];
          name   = m[4];
          sym = '.'.join(scope) + '.' + name;
          self.symbols[sym] = { 'type':type, 'ref':ref, 'length':length };
          self.refs[ref] = { 'type':type, 'length':length };
      elif (line[0] == "#"):
        self.dataSectionLineStart = self.linenumber - 1;
        return;
  def restart(self):
    self.time = None;
    self.linenumber = self.dataSectionLineStart;
    self.values = {};
    self.sampledValues = {};
    self.windowValues = [];
  def parse1Timescale(self):
    self.time = None;
    start = self.linenumber;
    for line in self.lines[start:]:
      self.linenumber += 1;
      if (line[0] == "$"):
        continue;
      if (line[0] == "#"):
        if (self.time == None):
          self.time = int(line[1:-1]);
          continue;
        else:
          self.linenumber -= 1;
          return True;
      if (line[0] == "b"):
        m = re.split(r'\s+', line);
        value = m[0][1:];
        ref   = m[1];
        padding = self.refs[ref]['length'] - len(value);
        self.values[ref] = value[0] * padding + value;
      else:
        value = line[0];
        ref   = line[1:-1];
        self.values[ref] = value;
    if (self.time == None):
      return True;
    else:
      return False;
  def parse1ClkCycle(self, clk, rsts = None):
    clkref  = self.symbols[clk['symbol']]['ref'];
    clkedge = '1' if clk['edge'] == "posedge" else '0';
    asyncrsts = {};
    syncrsts  = {};
    rstvalLast = {};
    if (rsts):
      for rst in rsts:
        rstref  = self.symbols[rst['symbol']]['ref'];
        rstedge = '1' if rst['edge'] == "posedge" or rst['edge'] == "high" else '0';
        if (rst['edge'] == "posedge" or rst['edge'] == "negedge"):
          asyncrsts[rstref] = rstedge;
        else:
          syncrsts[rstref] = rstedge;
        rstvalLast[rstref] = 'x';
    clkvalLast = 'x';
    while (self.parse1Timescale()):
      clkval = self.values[clkref];
      sample = False;
      if (clkvalLast != clkval):
        clkvalLast = clkval;
        if (clkval == clkedge):
          sample = True;
          for rstref in syncrsts.keys():
            rstval = self.values[rstref];
            if (rstval == syncrsts[rstref]):
              sample = False;
      for rstref in asyncrsts.keys():
        rstval = self.values[rstref];
        if (rstval != rstvalLast[rstref]):
          rstvalLast[rstref] = rstval;
          if (rstval == asyncrsts[rstref]):
            sample = True;
        if (rstval == asyncrsts[rstref]):
          sample = False;
      if (sample == True):
        return 1;
      import copy;
      self.sampledValues = copy.deepcopy(self.values);
    return 0;
  def parseWindow(self, windowSize, clk = None, rsts = None):
    if (len(self.windowValues) == windowSize):
        del self.windowValues[0];
    if (clk):
      while (self.parse1ClkCycle(clk, rsts)):
        import copy;
        self.windowValues.append(copy.deepcopy(self.sampledValues));
        if (len(self.windowValues) == windowSize):
          return 1;
    else:
      while (self.parse1Timescale()):
        import copy;
        self.windowValues.append(copy.deepcopy(self.values));
        if (len(self.windowValues) == windowSize):
          return 1;
    return 0;

if __name__ == "__main__":
  vcd = VcdParser("./case/arbiter.vcd");
  print("Symbols");
  for symbol in vcd.symbols.keys():
    print("\t" + symbol);
  print("Test");
  ref = vcd.symbols['top.U.rst']['ref'];
  while(vcd.parse1ClkCycle({'symbol':'top.U.clk','edge':'posedge'}, [{'symbol':'top.U.rst','edge':'high'}])):
    #print(vcd.time);
    print(vcd.sampledValues[ref]);

  vcd.restart();
  print("Test");
  while(vcd.parse1ClkCycle({'symbol':'top.U.clk','edge':'posedge'})):
    #print(vcd.time);
    print(vcd.sampledValues[ref]);
  
  vcd.restart();
  print("Test");
  while(vcd.parseWindow(3, {'symbol':'top.U.clk','edge':'posedge'})):
    #print(vcd.time);
    print(vcd.windowValues[1][ref]);
  vcd.restart();
  print("Test");
  while(vcd.parseWindow(3)):
    #print(vcd.time);
    print(vcd.windowValues[1][ref]);  

