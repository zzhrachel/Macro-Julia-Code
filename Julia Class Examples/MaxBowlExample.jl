#-----------------------------------------------------------------------------------------------------------------------------
# PURPOSE : Find the maximum of z = -0.5*(x-a)^2 -0.5*(y-b)^2 using Julia's Optim.jl package's optimize minimization routine:
#-----------------------------------------------------------------------------------------------------------------------------

using Optim
using PythonPlot

struct Parameter_Values
    a :: Real
    b :: Real
end

function MaxBowlExampleFun(x0 :: Array{Float64,1}, Param :: Parameter_Values)

    #--------------------------------------------------------------------------------------
    # PURPOSE: Solve for the values of (x,y) that maximize z = -1/2*(x-a)^2 - 1/2*(y-b)^2
    #--------------------------------------------------------------------------------------

    # Retrieve parameters from the structure "Param"
    a   = Param.a
    b   = Param.b
    # Retrieve current guess of optimal values of (x,y).
    x   = x0[1]
    y   = x0[2]

    # Construct current value of function
    zval = 0.5*(x-a)^2 + 0.5*(y-b)^2

    return zval
end

##   Define Parameter Values
a   = 5
b   = 7
Param = Parameter_Values(a,b)
# Alternatively we could have done the following but doing so would not have created the variables a and b with global scope:
# Param = Parameter_Values(5,7)

## Plot the function;
plotbowl = 0
if plotbowl == 1
    xmin = 0
    xmax = 10
    numx = 101
    ymin = 0
    ymax = 10
    numy = 101
    x   = range(xmin,stop=xmax,length=numx)
    y   = range(ymin,stop=ymax,length=numy)
    XMat = repeat(x,1,numy)
    YMat = repeat(y',numx,1)
    ZMat = -1/2*(XMat .- a).^2 - 1/2*(YMat .-b ).^2  # Must use .+ or .- (with spaces before and after) when adding constants to
                                                     # elements of an Array, Vector or Matrix.

    fig = figure("Bowl Plot")
    subplot(1,2,1, projection="3d")
    surf(XMat,YMat,ZMat,cmap = ColorMap("jet"))
    title("Bowl Plot")
    xlabel("x")
    ylabel("y")
    zlabel("z")
    axis([xmin, xmax, ymin, ymax])    
    ax = subplot(1,2,2)  # Doing a bunch of stuff to get the contour lines to display their levels on the contour lines.
    CS = ax.contour(XMat, YMat, ZMat, cmap=ColorMap("jet"))
    #contourf(XMat, YMat, ZMat, linewidth=2.0)
    xlabel("X")
    ylabel("Y")
    title("Contour Plot")
    tight_layout()
    ax.clabel(CS,inline=true,fontsize=12) # Label contours
    axis([xmin, xmax, ymin, ymax])
    sleep(2)
end


#######################################################
#                                                     #
#   Set Initial Guess at the Optimal Value of (x.y)   #
#                                                     #
#######################################################

# See documentation for the Optim.jl package.  

x0 = [3.0; 4.0]   # Initial guess at optimal value of x.
#xx = optimize(x->MaxBowlExampleFun(x,Param),x0)
min_results = optimize(x->MaxBowlExampleFun(x,Param),x0,BFGS())  # Use BFGS as the optimization algorithm
x = min_results.minimizer
f_min = min_results.minimum















