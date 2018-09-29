import numpy
import os

def run():
    file_list = os.listdir( './img_dir' )
    print( len(file_list) )
   
    out1_list = [ s for s in file_list if s.find( 'out1' )!= -1 ]
    out2_list = [ s for s in file_list if s.find( 'out2' )!= -1 ]

    #for file in out1_list:
    #    os.remove( os.path.join( os.getcwd() ,'img_dir' , file ) )
    #for file in out2_list:
    #    os.remove( os.path.join( os.getcwd() , 'img_dir', file ) )


run()
