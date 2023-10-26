# Carabao Utilities:
# - class Caracow
# - class Carabull
# - class Carabao
# - class Carashit
#
# Try:
#    from carabao import *
#
#    help(Carabao)         # help on Carabao class in general
#    help(Carabao.pack)    # help on Carabao's pack method
#    o = Carashit()
#    o.shit

#===============================================================================
# class Caracow (one of Carabao's basic classes)
#===============================================================================

from ypstruct import struct

class Caracow:
    """
    CARACOW: one of the two basis classes of Carabao class

                o = Caracow()     % default construction
                o = Caracow(tag)  % construct with specific tag
                oo = Caracow(o)   % copy constructor
    """
    def __init__(self,arg=None):
        tag = 'Caracow'
        if arg == None:
            arg = 'shell'

            # convert a Caracow object into a structure

        if type(arg).__name__ == 'Caracow':
            arg = arg.pack()

            # dispatch on argument class

        if type(arg).__name__ == 'str':
            self.tag = tag
            self.type = arg
            self.par = struct()
            self.data = ()
            self.work = struct()
        elif type(arg).__name__ == 'struct':
            self.tag = arg.tag
            self.type = arg.type
            self.par = arg.par
            self.data = arg.data
            self.work = arg.work
        else:
            raise Exception('bad argument class (arg1)!')

    def __repr__(self):
        return "Caracow()"

    def __str__(self):
        return "<Caracow %s>" % self.type

    def pack(self):
        """
        PACK:  pack Caracow object into a structure

               see also: Caracow
        """
        return struct(type=self.type, tag=self.tag, par=self.par,
                      data=self.data, work=self.work)


#===============================================================================
# class Carashit
#===============================================================================

class Carashit:
    def __init__(self):
        self.shit = '*** bullshit ***'

#===============================================================================
# setup neural playground
#===============================================================================
"""
print("setting up neural playground ...")

import carabao
import carabao.screen
import carabao.cell

import importlib
importlib.reload(carabao.screen)  # reload module
importlib.reload(carabao.cell)    # reload module

print("... packages reloaded")
"""
