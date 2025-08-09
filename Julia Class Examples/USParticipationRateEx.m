clear all

%% Plot the Unemployment Rate

% Get Data:
sheetnum = 1;
filename = 'USParticipationRate.xlsx';
[ndata,text,alldata] = xlsread(filename,sheetnum);
% Read data from sheet number "sheetnum".  "alldata" will be an array with text
% in the unprocessed data from the sheet.  "ndata" is an array with only the
% numeric data from the sheet.  "text" is an array with the text data from the
% Excel spreadsheet.

% See "alldata" to see the excel spreadsheet data with headings!
pr  = ndata(:,3);   % Create varaiable ur by taking all the observations from third
                    % column of the matrix ndata (unemployment rates)

% Observations from the beginning of the financial crisis starts in August, 2007.
% Thus we want to plot data before and after row 81 of the matrix ndata as separate
% lines.

nn = DateNumTicks(1948,2015,[1,7,1]);  % See DateNumTicks.m for notes about usage.

fs  = 16;
figure
plot(nn,pr);
datetick('x',12);
set(gca,'xlim',[min(nn), max(nn)]);
xlabel('Date','fontname','times','fontsize',fs);
title('US Labour Force Participation Rate (16+ years old)','fontname','times','fontsize',fs);