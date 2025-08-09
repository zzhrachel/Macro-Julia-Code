# Use fcsolve.jl to solve optimization problems.  It is a modified BFGS algorithm.  Does get stuck if there are many cliffs and ragged surface.
using Distributions
using Printf
using LinearAlgebra


# Use if solving univariate case:
function fcsolve(FUN :: Function, x :: Real, options :: Array, VarArgIn...)
    # Solves for FUN(x,P1,P2,...)=0
    #
    # FUN should be written -   so that parametric arguments are packed in to vectors P1,...
    #                       -   so that if presented with a matrix x, it produces a return value of
    #                           same dimension of x.  
    #
    # rc = 0 if convergence is OK
    #      4 is the maximum number of iterations has been reached
    #
    # option is an optional vector to control the algorithm:
    #
    # itmax : maximum number of iterations (sum of abs. values small enough to be a solution) (1000)
    # crit  : stopping criterion (1e-9)
    # delta : differencing interval for numerical gradient (1e-8)
    # alpha : tolerance on rate of descent (1e-3)
    # dispo : Displaystyle (0 nothing, 1 minimum (default), 2 full
    #
    # (c) Chris Sims 1997 revised by F. Collard (1999)

    if ~isempty(options)
       itmax=options[1]   
       crit =options[2]
       delta=options[3]
       alpha=options[4]
       dispo=options[5] 
    else
       itmax=1000       # max no. of iterations    
       crit=1e-6        # sum of abs. values small enough to be a solution
       delta=1e-8       # differencing interval for numerical gradient
       alpha=1e-3       # tolerance on rate of descent
       #dispo=0         # no display
       dispo=1          # partial display
    end
    nv   = length(x) # in this case, as x0 is a Real number rather than a Vector (Array{Float64,1}), nv = 1.
    tvec = delta
    done = 0;

    # VarArgIn = VarArgIn[1]
    # f0 = FUN(x,VarArgIn)
    f0   = FUN(x,VarArgIn...)

    af0  = sum(abs(f0))
    af00 = 0.0
    af00 = af0
    itct = 0
    grad = 0
    rc   = 0
    while done == 0
        if itct>3 && af00-af0<crit*max(1,af0) && rem(itct,2)==1
           randomize=1;
        else
            for i in 1 : nv
                #grad[1:nv,i] .= (FUN(x.+tvec[:,i],VarArgIn...)-FUN(x,VarArgIn...))/delta  # Don't use this as x0 is a Real type.
                grad = (FUN(x+tvec,VarArgIn...)-FUN(x,VarArgIn...))/delta
            end
            if isreal(grad)
                if 1/cond(grad,1) < 1e-12
                    grad=grad+tvec
                end
                dx0 = -grad\f0
                randomize=0
            else
                println("imaginary gradient")
                randomize = 1
            end
        end
        if randomize == 1
            distx   = Normal(0,1)
            println("Random Search")
            dx0=norm(x)/rand(distx)
        end
        lambda  = 1
        lambdamin = 1
        fmin    = f0
        xmin    = x
        afmin   = af0
        dxSize  = norm(dx0)
        factor  = 0.6
        shrink  = 1
        subDone = 0
        f2      = 0
        while subDone == 0
            dx = lambda*dx0;
            f  = FUN(x+dx,VarArgIn...)
            af = sum(abs(f))
            f2 = sum(f^2)
            if af < afmin
                afmin = af
                fmin = f
                lambdamin = lambda
                xmin = x + dx
            end
            if ((lambda >0) && (af0-af < alpha*lambda*af0)) || ((lambda<0) && (af0-af < 0))
                if ~shrink == 1
                    factor = factor^0.6
                    shrink = 1
                end
                if abs(lambda*(1-factor))*dxSize > 0.1*delta
                    lambda = factor*lambda
                elseif (lambda > 0) && (factor == 0.6) # i.e., we've only been shrinking
                    lambda = -0.3
                else
                    subDone = 1
                    if lambda > 0
                        if factor == 0.6
                            rc = 2
                        else
                            rc = 1
                        end
                    else
                        rc = 3
                    end
                end
            elseif (lambda >0) && (af-af0 > (1-alpha)*lambda*af0)
                if shrink == 1
                    factor = factor^0.6
                    shrink = 0
                end
                lambda = lambda/factor
            else # good value found
                subDone = 1
                rc = 0
            end
        end
        itct = itct + 1
        if dispo == 1
            @printf("iteration : %2.0f, Sum |f| : %e, ||f||² : %e, step : %e, conv(lambda) %d",itct,afmin,f2,lambdamin,rc)
            println("")
        end
        x    = xmin
        f0   = fmin
        af00 = af0
        af0  = afmin
        if itct >= itmax
            done = 1
            rc   = 4
        elseif af0 < crit
            done = 1
            rc   = 0
        end
    end

    if dispo == 1 
        println("  ")
        println("-----------------------------------------------------")
        println("                     RESULTS")
        println("-----------------------------------------------------")
        println(" ")
        if rc == 0
            println("Convergence achieved properly")
        elseif rc == 4
            println("Maximal number of iterations reached")
        else
            println("Convergence not achieved properly, try another starting value")
        end
        println(" ") 
        @printf("Iteration n          : %5.0f",itct)
        #@printf("Elapsed Time (in sec.): %5.2f',etime(clock,h1))) 
        println(" ")
        println("Solution             :") 
        println(x)
        #@printf("\n %15.6f",x)
        println(" " )
        println("-----------------------------------------------------")
    end
    return x
end

# Use if solving multivariate case:
function fcsolve(FUN :: Function, x :: Vector{Real}, options :: Array, VarArgIn...)
	# Solves for FUN(x,P1,P2,...)=0
	#
	# FUN should be written - 	so that parametric arguments are packed in to vectors P1,...
	#	 					- 	so that if presented with a matrix x, it produces a return value of
	#	 						same dimension of x.  
	#
	# rc = 0 if convergence is OK
	#      4 is the maximum number of iterations has been reached
	#
	# option is an optional vector to control the algorithm:
	#
	# itmax : maximum number of iterations (sum of abs. values small enough to be a solution) (1000)
	# crit  : stopping criterion (1e-9)
	# delta : differencing interval for numerical gradient (1e-8)
	# alpha : tolerance on rate of descent (1e-3)
	# dispo : Displaystyle (0 nothing, 1 minimum (default), 2 full
	#
	# (c) Chris Sims 1997 revised by F. Collard (1999)

	if ~isempty(options)
	   itmax=options[1]   
	   crit =options[2]
	   delta=options[3]
	   alpha=options[4]
	   dispo=options[5] 
	else
	   itmax=1000       # max no. of iterations    
	   crit=1e-6		# sum of abs. values small enough to be a solution
	   delta=1e-8		# differencing interval for numerical gradient
	   alpha=1e-3		# tolerance on rate of descent
	   #dispo=0         # no display
	   dispo=1			# partial display
	end
	nv   = length(x);
	tvec = delta*Matrix{Float64}(I,nv,nv);
	done = 0;

	# VarArgIn = VarArgIn[1]
	# f0 = FUN(x,VarArgIn)
	f0 	 = FUN(x,VarArgIn...)

	af0  = sum(abs.(f0))
	af00 = 0.0
	af00 = af0
	itct = 0
	grad = zeros(nv,nv)
	rc   = 0
	while done == 0
	    if itct>3 && af00-af0<crit*max(1,af0) && rem(itct,2)==1
	       randomize=1;
	    else
	        for i in 1 : nv
	            # grad[1:nv,i] = (FUN(x+tvec[:,i],VarArgIn)-FUN(x,VarArgIn))/delta
	            grad[1:nv,i] = (FUN(x+tvec[:,i],VarArgIn...)-FUN(x,VarArgIn...))/delta
	        end
	        if isreal(grad)
	          	if 1/cond(grad,1) < 1e-12
	            	grad=grad+tvec;
	         	end
	         	dx0 = -grad\f0
	         	randomize=0
	      	else
	        	println("imaginary gradient")
	        	randomize = 1
	      	end
	   	end
	   	if randomize == 1
	   		distx	= Normal(0,1)
	    	println("Random Search")
	        dx0=norm(x)./rand(distx,size(x,1));
	    end
	    lambda  = 1
	    lambdamin = 1
	    fmin	= f0
	    xmin    = x
	    afmin   = af0
	    dxSize  = norm(dx0)
	    factor  = 0.6
	    shrink  = 1
	    subDone = 0
	    f2		= 0
	    while subDone == 0
	        dx = lambda*dx0;
			# f  = FUN(x+dx,VarArgIn)
	        f  = FUN(x+dx,VarArgIn...)
	        af = sum(abs.(f))
	        f2 = sum(f.^2)
	        if af < afmin
	         	afmin = af
	         	fmin = f
	         	lambdamin = lambda
	         	xmin = x + dx
	      	end
	        if ((lambda >0) && (af0-af < alpha*lambda*af0)) || ((lambda<0) && (af0-af < 0))
	        	if ~shrink == 1
	            	factor = factor^0.6
	            	shrink = 1
	         	end
	         	if abs.(lambda*(1-factor))*dxSize > 0.1*delta
	            	lambda = factor*lambda
	        	elseif (lambda > 0) && (factor == 0.6) # i.e., we've only been shrinking
	            	lambda = -0.3
	         	else
	            	subDone = 1
	            	if lambda > 0
	               		if factor == 0.6
	                  		rc = 2
	                	else
	                    	rc = 1
	               		end
	            	else
	                	rc = 3
	            	end
	            end
	        elseif (lambda >0) && (af-af0 > (1-alpha)*lambda*af0)
	        	if shrink == 1
	            	factor = factor^0.6
	            	shrink = 0
	         	end
	         	lambda = lambda/factor
	      	else # good value found
	        	subDone = 1
	         	rc = 0
	      	end
	   	end
	    itct = itct + 1
	    if dispo == 1
		    @printf("iteration : %2.0f, Sum |f| : %e, ||f||² : %e, step : %e, conv(lambda) %d",itct,afmin,f2,lambdamin,rc)
		    println("")
	    end
	   	x 	 = xmin
	   	f0   = fmin
	   	af00 = af0
	   	af0  = afmin
	   	if itct >= itmax
	   		done = 1
	   	   	rc 	 = 4
	   	elseif af0 < crit
	   		done = 1
	   	   	rc   = 0
	   	end
    end

    if dispo == 1 
		println("  ")
		println("-----------------------------------------------------")
		println("                     RESULTS")
		println("-----------------------------------------------------")
		println(" ")
		if rc == 0
		    println("Convergence achieved properly")
		elseif rc == 4
			println("Maximal number of iterations reached")
		else
			println("Convergence not achieved properly, try another starting value")
		end
		println(" ") 
		@printf("Iteration n          : %5.0f",itct)
		#@printf("Elapsed Time (in sec.): %5.2f',etime(clock,h1))) 
		println(" ")
		println("Solution             :") 
		println(x)
		#@printf("\n %15.6f",x)
		println(" " )
		println("-----------------------------------------------------")
	end
    return x
end

# Use if solving multivariate case:
function fcsolve(FUN :: Function, x :: Vector{Int64}, options :: Array, VarArgIn...)
	# Solves for FUN(x,P1,P2,...)=0
	#
	# FUN should be written - 	so that parametric arguments are packed in to vectors P1,...
	#	 					- 	so that if presented with a matrix x, it produces a return value of
	#	 						same dimension of x.  
	#
	# rc = 0 if convergence is OK
	#      4 is the maximum number of iterations has been reached
	#
	# option is an optional vector to control the algorithm:
	#
	# itmax : maximum number of iterations (sum of abs. values small enough to be a solution) (1000)
	# crit  : stopping criterion (1e-9)
	# delta : differencing interval for numerical gradient (1e-8)
	# alpha : tolerance on rate of descent (1e-3)
	# dispo : Displaystyle (0 nothing, 1 minimum (default), 2 full
	#
	# (c) Chris Sims 1997 revised by F. Collard (1999)

	if ~isempty(options)
	   itmax=options[1]   
	   crit =options[2]
	   delta=options[3]
	   alpha=options[4]
	   dispo=options[5] 
	else
	   itmax=1000       # max no. of iterations    
	   crit=1e-6		# sum of abs. values small enough to be a solution
	   delta=1e-8		# differencing interval for numerical gradient
	   alpha=1e-3		# tolerance on rate of descent
	   #dispo=0         # no display
	   dispo=1			# partial display
	end
	nv   = length(x);
	tvec = delta*Matrix{Float64}(I,nv,nv);
	done = 0;

	# VarArgIn = VarArgIn[1]
	# f0 = FUN(x,VarArgIn)
	f0 	 = FUN(x,VarArgIn...)

	af0  = sum(abs.(f0))
	af00 = 0.0
	af00 = af0
	itct = 0
	grad = zeros(nv,nv)
	rc   = 0
	while done == 0
	    if itct>3 && af00-af0<crit*max(1,af0) && rem(itct,2)==1
	       randomize=1;
	    else
	        for i in 1 : nv
	            # grad[1:nv,i] = (FUN(x+tvec[:,i],VarArgIn)-FUN(x,VarArgIn))/delta
	            grad[1:nv,i] = (FUN(x+tvec[:,i],VarArgIn...)-FUN(x,VarArgIn...))/delta
	        end
	        if isreal(grad)
	          	if 1/cond(grad,1) < 1e-12
	            	grad=grad+tvec;
	         	end
	         	dx0 = -grad\f0
	         	randomize=0
	      	else
	        	println("imaginary gradient")
	        	randomize = 1
	      	end
	   	end
	   	if randomize == 1
	   		distx	= Normal(0,1)
	    	println("Random Search")
	        dx0=norm(x)./rand(distx,size(x,1));
	    end
	    lambda  = 1
	    lambdamin = 1
	    fmin	= f0
	    xmin    = x
	    afmin   = af0
	    dxSize  = norm(dx0)
	    factor  = 0.6
	    shrink  = 1
	    subDone = 0
	    f2		= 0
	    while subDone == 0
	        dx = lambda*dx0;
			# f  = FUN(x+dx,VarArgIn)
	        f  = FUN(x+dx,VarArgIn...)
	        af = sum(abs.(f))
	        f2 = sum(f.^2)
	        if af < afmin
	         	afmin = af
	         	fmin = f
	         	lambdamin = lambda
	         	xmin = x + dx
	      	end
	        if ((lambda >0) && (af0-af < alpha*lambda*af0)) || ((lambda<0) && (af0-af < 0))
	        	if ~shrink == 1
	            	factor = factor^0.6
	            	shrink = 1
	         	end
	         	if abs.(lambda*(1-factor))*dxSize > 0.1*delta
	            	lambda = factor*lambda
	        	elseif (lambda > 0) && (factor == 0.6) # i.e., we've only been shrinking
	            	lambda = -0.3
	         	else
	            	subDone = 1
	            	if lambda > 0
	               		if factor == 0.6
	                  		rc = 2
	                	else
	                    	rc = 1
	               		end
	            	else
	                	rc = 3
	            	end
	            end
	        elseif (lambda >0) && (af-af0 > (1-alpha)*lambda*af0)
	        	if shrink == 1
	            	factor = factor^0.6
	            	shrink = 0
	         	end
	         	lambda = lambda/factor
	      	else # good value found
	        	subDone = 1
	         	rc = 0
	      	end
	   	end
	    itct = itct + 1
	    if dispo == 1
		    @printf("iteration : %2.0f, Sum |f| : %e, ||f||² : %e, step : %e, conv(lambda) %d",itct,afmin,f2,lambdamin,rc)
		    println("")
	    end
	   	x 	 = xmin
	   	f0   = fmin
	   	af00 = af0
	   	af0  = afmin
	   	if itct >= itmax
	   		done = 1
	   	   	rc 	 = 4
	   	elseif af0 < crit
	   		done = 1
	   	   	rc   = 0
	   	end
    end

    if dispo == 1 
		println("  ")
		println("-----------------------------------------------------")
		println("                     RESULTS")
		println("-----------------------------------------------------")
		println(" ")
		if rc == 0
		    println("Convergence achieved properly")
		elseif rc == 4
			println("Maximal number of iterations reached")
		else
			println("Convergence not achieved properly, try another starting value")
		end
		println(" ") 
		@printf("Iteration n          : %5.0f",itct)
		#@printf("Elapsed Time (in sec.): %5.2f',etime(clock,h1))) 
		println(" ")
		println("Solution             :") 
		println(x)
		#@printf("\n %15.6f",x)
		println(" " )
		println("-----------------------------------------------------")
	end
    return x
end
