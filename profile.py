# Add some self profiling information 
# Allow nested timer exists

import time

class TimerError(Exception):
    """Exception in ProfTimer class"""

class ProfTimer:
    def __init__(self, timername="Timer for DLRM Activity"):
        self._name = timername
        self._start = 0.0
        self._count = 0
        self._elapsed = 0.0

    def start(self):
        """Start a new timer"""
        self._start = time.perf_counter()

    def stop(self):
        if self._start == 0.0:
            raise TimerError(f"Timer is not running.")
        self._elapsed += time.perf_counter() - self._start
        self._count += 1
        self._start = 0.0
 
    def count(self):
        return _self._count

    def reset(self):
        self._elapsed = 0.0
        self._count = 0

    def elapsed(self):
        return self._elapsed 

    def output(self, level):
        if level == 0:
            print(f"{self._name }: {self._elapsed:0.6f} seconds with counts {self._count}")
        else:
            print(f"    {self._name }: {self._elapsed:0.6f} seconds with counts {self._count}")
   
alltimers = []
tmGetData = ProfTimer("GetData")
tmFwd     = ProfTimer("Forword")
tmLoss    = ProfTimer("Loss   ")
tmZero    = ProfTimer("Zero   ")
tmBwd     = ProfTimer("Backwrd")
tmOpt     = ProfTimer("Opt    ")
tmSync    = ProfTimer("CudaSyn")
tmSync1   = ProfTimer("CudaSy1")
tmSync2   = ProfTimer("CudaSy2")
tmSync3   = ProfTimer("CudaSy3")

tmH2D     = ProfTimer("CopyH2D")
tmEmb     = ProfTimer("EMB    ")
tmA2A     = ProfTimer("All2All")
tmA2A1    = ProfTimer("All2All1")
tmBot     = ProfTimer("Bottom ")
tmInt     = ProfTimer("Inter  ")
tmTop     = ProfTimer("Top MLP")
tmAllGa   = ProfTimer("Allgath")

tmA2A10    = ProfTimer("All2All10")
tmA2A11    = ProfTimer("All2All11")
tmA2A12    = ProfTimer("All2All12")
tmA2A13    = ProfTimer("All2All13")

def tmClear():
        
    tmGetData.reset()
    tmFwd.reset()
    tmLoss.reset()
    tmZero.reset()
    tmBwd.reset()
    tmOpt.reset()
    tmSync.reset()
    tmSync1.reset()
    tmSync2.reset()
    tmSync3.reset()

    tmH2D.reset()
    tmEmb.reset()
    tmA2A.reset()
    tmA2A1.reset()
    tmBot.reset()
    tmInt.reset()
    tmTop.reset()
    tmAllGa.reset()

    tmA2A10.reset()
    tmA2A11.reset()
    tmA2A12.reset()
    tmA2A13.reset()

def tmSummary(pid):
    
    print("Summary of the tm timers:")
    print("---------{:6d}----------------".format(pid))
    tmGetData.output(0)
    tmFwd.output(0)
    tmH2D.output(1)
    tmEmb.output(1)
    tmA2A.output(1)
    tmA2A1.output(1)
    tmBot.output(1)
    tmInt.output(1)
    tmTop.output(1)
    tmAllGa.output(1)
    tmLoss.output(0)
    tmZero.output(0)
    tmBwd.output(0)
    tmOpt.output(0)
#    tmSync.output(0)
    tmSync1.output(0)
    tmSync2.output(0)
    tmSync3.output(0)

    tmA2A10.output(1)
    tmA2A11.output(1)
    tmA2A12.output(1)
    tmA2A13.output(1)
    print("========={:6d}================".format(pid))

if __name__ == "__main__":
    
    t1 = ProfTimer("Test1")
    t1.start()
    time.sleep(3)
    t1.stop()
    t1.elapsed()
    t1.output()

    t1.start()
    time.sleep(5)
    t1.stop()
    t1.output()


