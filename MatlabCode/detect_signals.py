__author__ = "Sergi Sancho, Adriana Fernandez, Eric Lopez y Gerard Marti"
__credits__ = ['Sergi Sancho', 'Adriana Fernandez', 'Eric Lopez', 'Gerard Marti']
__license__ = "GPL"
__version__ = "1.0"

import matlab.engine

def run():
    eng = matlab.engine.start_matlab()
    eng.addpath('MatlabCode')
    eng.Start(nargout=0)



if __name__ == '__main__':
    run()