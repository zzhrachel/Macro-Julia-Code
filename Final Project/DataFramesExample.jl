using DataFrames
using Dates      # To create date ticks in plots
using PythonPlot
using XLSX
include("monthlytoquarterly.jl")
include("bpass.jl")

# Import tables from Excel worksheets included in the Excel .xlsx workbook.  Create DataFrames from each worksheet.  A DataFrame type looks like a Stata display.
df_Barnichon = DataFrame(XLSX.readtable("DMPProjectData.xlsx","Barnichon Monthly Data"))
df_EMR = DataFrame(XLSX.readtable("DMPProjectData.xlsx","Ritter Data"))
df_Productivity = DataFrame(XLSX.readtable("DMPProjectData.xlsx","Output Per Hour"))
df_JobData = DataFrame(XLSX.readtable("DMPProjectData.xlsx","Job Filling vs Job Finding"))


ExampleNumber = 2
if ExampleNumber == 1
    # Estimate matching function shocks using Ritter u-e probabilities and Barnichon's
    # measures of vacancies and unemployment.  Data runs from January 1968 through
    # the end of 2012.  Ritter's data is quarterly and Barnichon's data is monthly so
    # take the 3 month averages of Barnichon's data to produce quarterly data.


	# Pull out unemployment and vacancy data from Barnichon data:
	um = df_Barnichon[:,2]
	vm = df_Barnichon[:,3]
	# Average monthly unemployment observations into quarterly observations
	uq	= monthlytoquarterly(um)


	# Pull out the transition probability data from u to e (from Elsby, Michaels and Ratner (JEL))
	pue = df_EMR[:,2]   

	# Uset the bandpass filter to extra cyclical fluctuations from monthly and quarterly unemployment data:

	bpum = bpass(log.(um),6,240) # Filter monthly data to rid of periodicities longer than 20 years and shorter than 6 months
	bpuq = bpass(log.(uq),2,96)  # Filter monthly data to rid of periodicities longer than 20 years and shorter than 2 quarters.

    nnm  = collect(Date(1968,1,1):Dates.Month(1):Date(2012,12,1))
    nnq  = collect(Date(1968,1,1):Dates.Month(3):Date(2012,12,1))
	
	figure
	subplot(1,2,1)
	plot(nnm,um,color="b",label="Raw Data")
	plot(nnm,bpum,color="r",label="BP Filtered")
	xlabel("Date",fontname="times",fontsize="14")
	legend(loc=1)
	title("Monthly Unemployment Rate Data (Raw vs Filtered)",fontname="times",fontsize="14")
	subplot(1,2,2)
	plot(nnq,uq,color="b",label="Raw Data")
	plot(nnq,bpuq,color="r",label="BP Filtered")
	xlabel("Date",fontname="times",fontsize="14")
	legend(loc=1)
	title("Quarterly Unemployment Rate Data (Raw vs Filtered)",fontname="times",fontsize="14")
else
	## Plot Raw and Filtered Labour Productivity Data.

	lpl  = df_Productivity[:,4] 
	lp   = bpass(log.(lpl),2,80)

    nnq  = collect(Date(1968,1,1):Dates.Month(3):Date(2012,12,1))

	figure;
	subplot(1,2,1)
	plot(nnq,lpl,color="b")
	xlabel("Date",fontname="times",fontsize="14")
	title("Real Output per Hour Index: Raw Data",fontname="times",fontsize="14")
	subplot(1,2,2)
	plot(nnq,lp,color="r")
	xlabel("Date",fontname="times",fontsize="14")
	ylabel("% Deviation from Trend",fontname="times",fontsize="14")
	title("Real Output per Hour Index: Filtered Data",fontname="times",fontsize="14")	
end

println("All Done")