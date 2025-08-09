## PURPOSE : An example of using functions and composite types (sometimes known in other languages as structures).

struct Parameter_Values
    # This defines a composite type named Parameter_Values which stores the variable a (an integer type), b (a float64 type)
    # and c (an integer type).   If we try to store c as a number with decimals we will get an error message because it is 
    # assumed that c is bound to an integer.
    a :: Int64
    b :: Float64
    c :: Int64
end

# struct Parameter_Values
#     # If we want to allow all a, b, and c to be either be integers or real numbers with decimals then we can do the following:
#     a :: Real
#     b :: Real
#     c :: Real
# end

function FunctionExample1(x0 :: Vector, Param :: Parameter_Values)
    # Retrieve parameters:
    a   = Param.a
    b   = Param.b
    c   = Param.c

    x   = x0[1]
    y   = x0[2]
    z   = x0[3]

    f = sin(x) + a*x*y*z^2
    g = b*log(x)*log(y)
    h = sqrt(x*y) + (z^3)/c

    Yly = 26

    return f, g, h
end

function FunctionExample2(Param :: Parameter_Values)
    # Retrieve parameters:
    a   = Param.a
    b   = Param.b
    c   = Param.c

    f = sin(x) + a*x*y*z^2
    g = b*log(x)*log(y)
    h = sqrt(x*y) + (z^3)/c

    return f, g, h
end

# Define parameters:
a = 2   
b = 2.5
c = -3

Param = Parameter_Values(a,b,c) # This stores the values of a, b, and c in the composite type/structure Param which
                                # is a ``Parameter_Values type.

x = 2
y = 1
z = 4

funcopt = 2
if funcopt == 0
    # Without using a function:
    f = sin(x) + a*x*y*z^2;
    g = b*log(x)*log(y);
    h = sqrt(x*y) + (z^3)/c;

    display([f ; g ; h]);
elseif funcopt == 1
    x0  = [x ; y ; z]
    f, g, h = FunctionExample1(x0,Param)
    display([f,g,h])
    # display(Yly) # Executing this line will result in an error message because Yly was created inside the function and not returned as
                   # part of the function's output.  As such, it is a variable with local scope - it can be used locally within the function
                   # but not outside of it.  On the other hand, x, y, and z are global in scope and can be used within a function without
                   # passing their values into the function as an input (see example by setting funcopt=2)
else
    f, g, h = FunctionExample2(Param)
    display([f,g,h])
end

println("All Done!")
