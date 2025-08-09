
#--------------------------------------------------------------------------------------------
#  JULIA COMMAND FOR BAND PASS FILTER ADAPTED FROM MATLAB CODE: fX = bpass(X,pl,pu)
#
#     This is a Matlab program that filters time series data using an approximation to the 
#     band pass filter as discussed in the paper "The Band Pass Filter" by Lawrence J. 
#     Christiano and Terry J. Fitzgerald (1999).
#
#  Required Inputs:
#  X     - series of data (T x 1)
#  pl    - minimum period of oscillation of desired component 
#  pu    - maximum period of oscillation of desired component (2<=pl<pu<infinity).
#   
#  Output:
#  fX - matrix (T x 1) containing filtered data 
#   
#   Examples: 
#     Quarterly data: pl=6, pu=32 returns component with periods between 1.5 and 8 yrs.
#     Monthly data:   pl=2, pu=24 returns component with all periods less than 2 yrs.
#
#  Note:  When feasible, we recommend dropping 2 years of data from the beginning 
#            and end of the filtered data series.  These data are relatively poorly
#            estimated.
#
#     ===============================================================================
#     This program contains only the default filter recommended in Christiano and 
#     Fitzgerald (1999). This program was written by Eduard Pelz and any errors are 
#     my own and not the authors. For those who wish to use optimal band-pass filters
#     other than the default filter please use the Matlab version of code available 
#     at www.clev.frb.org/Research/workpaper/1999/index.htm next to working paper 9906.
#     ===============================================================================
#
#     Version Date: 2/11/00 (Please notify Eduard Pelz at eduard.pelz@clev.frb.org or 
#     (216)579-2063 if you encounter bugs).  
#
# Adapted to Julia by Jake on Oct 5, 2016.
#--------------------------------------------------------------------------------------------

function bpass(X :: Array, pl :: Int64, pu :: Int64)
   if pu <= pl
      println("(bpass): pu must be larger than pl")
   end
   if pl < 2
      println("(bpass): pl less than 2 , reset to 2")
      pl = 2
   end

   T     = size(X,1)
   nvars = size(X,2)

   #  This section removes the drift from a time series using the 
   #	formula: drift = (X(T) - X(1)) / (T-1).                     
   #
   undrift = 1
   j = collect(1:T)'
   if undrift == 1
      drift = (X[T,1]-X[1,1])/(T-1)
      Xun = X - collect((j' .- 1)*drift)
   else
      Xun = X
   end

   #Create the ideal B's then construct the AA matrix
   b  = 2*pi/pl
   a  = 2*pi/pu
   bnot = (b-a)/pi
   bhat = bnot/2

   Bt = (sin.(j*b)-sin.(j*a))./(j*pi)
   B  = Bt'
   B[2:T,1] = B[1:T-1,1]
   B[1,1] = bnot

   AA = zeros(2*T,2*T)

   for i in 1 : T
      AA[i,i:i+T-1] = B'
      AA[i:i+T-1,i] = B   
   end
   AA = AA[1:T,1:T]

   AA[1,1] = bhat
   AA[T,T] = bhat

   for i in 1 : T-1
      AA[i+1,1] = AA[i,1] - B[i,1]
   	AA[T-i,T] = AA[i,1] - B[i,1]
   end

   #Filter data using AA matrix

   fX = AA*Xun
   
   return fX
end
