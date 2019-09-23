#!/usr/bin/env python
# coding: utf-8

# In[1]:


import panel
import panel as pn
pn.extension('plotly')
import pandas as pd


# UNIVARIATE ANALYSIS 

# In[2]:


data=pd.read_csv("F:/SCADA_Data/work/KSEB14-18.csv")
#wea_data=pd.read_csv("F:/SCADA_Data/work/Demand+Weather2015-18.csv")
data["date"]=pd.to_datetime(data['timestamp']).dt.date
data["month"]=pd.to_datetime(data['timestamp']).dt.month
data["hour"]=pd.to_datetime(data['timestamp']).dt.hour


# In[3]:


idx = data.groupby(['date'])['demand'].transform(max) == data['demand']
peak=data[idx]


# In[299]:


peak.head()


# In[296]:


peak['hour']=[str(h) for h in peak['hour']]


# In[298]:



peak['hour']=[datetime.strftime(datetime.strptime(t, '%H'), '%I %p') for t in peak['hour']]


# GRAPH 1

# In[300]:


import plotly.express as px


# In[302]:


#histogram-peak
fig0 = px.histogram(peak, x="hour",title='Histogram of demand consumption at peak hours')
plotly_pane0 = pn.pane.Plotly(fig0)
plotly_pane0


# In[9]:


dailymean=pd.DataFrame(data.groupby('date',as_index=False).agg({'demand':'mean'}))
dailymean["month"]=pd.to_datetime(dailymean['date']).dt.month


# In[10]:


dailymean.head()


# GRAPH 2

# In[303]:


#histogram-avg
fig1 = px.histogram(dailymean, x="demand",title="Histogram of demand")
#fig1.show()
plotly_pane1 = pn.pane.Plotly(fig1)
plotly_pane1


# GRAPH 3

# In[150]:


#monthly avg demand
y=[2014,2015,2016,2017,2018]
colors=['deepskyblue','red','darkgreen','grey','magenta']
import plotly.graph_objects as go
fig2 = go.Figure()
d1 = 0
d2 = 365
for i in range(5):
    fig2.add_trace(go.Scatter(
                x=(dailymean.iloc[d1:d2,:]).date,
                y=(dailymean.iloc[d1:d2,:]).demand,
                name=str(y[i]),
                line_color=colors[i],
               opacity=0.8))
    
    if y[i]%4 ==0:
        d1=d1+366
        d2=d2+366
    else:
        d1=d1+365
        d2=d2+365
        
fig2.update_layout(xaxis_range=['2014-01-01','2018-12-31'])
plotly_pane2 = pn.pane.Plotly(fig2)
plotly_pane2


# In[13]:


data['Year']=[ts.year for ts in data['date']]
dat_new=pd.DataFrame(data.groupby(['month','Year'],as_index=False).agg({'demand':'mean'}))
dat_new.sort_values(by=['Year','month'], inplace=True)
dat_new['Year']=[str(y) for y in dat_new['Year']]
import calendar
dat_new['month']=[calendar.month_abbr[m] for m in dat_new['month']]
dat_new['Month']=dat_new['month']+' '+dat_new['Year']
dat_new.drop(['month'],axis=1,inplace=True)


# In[14]:


x=dat_new['Year'].unique()
dat_new.iloc[1,:].Year


# In[15]:


from collections import OrderedDict
f=OrderedDict()
for y in x:
    f[y]=dat_new[dat_new['Year']==y]


# In[16]:


colors=['deepskyblue','red','darkgreen','grey','magenta']
for i in range(2):
    print(pd.DataFrame(f[x[i]]))


# GRAPH 4

# In[304]:


#monthly avg demand
import plotly.graph_objects as go
fig3 = go.Figure()
for i in range(5):
    fig3.add_trace(go.Scatter(
                x=(pd.DataFrame(f[x[i]])).Month,
                y=(pd.DataFrame(f[x[i]])).demand,
                name=x[i],
                line_color=colors[i],
               opacity=0.8))
fig3.update_layout(xaxis_range=['2014-01-01','2018-12-31'],title="Monthly load curve ")
#fig.show()
plotly_pane3 = pn.pane.Plotly(fig3)
plotly_pane3


# In[18]:


from pandas import Series
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
fig = go.Figure()


# MULTIVARIATE ANALYSIS

# In[45]:


wea_data=pd.read_csv("F:/SCADA_Data/work/Demand+Weather.csv")


# In[46]:


wea_data["date"]=pd.to_datetime(wea_data['Timestamp']).dt.date
wea_data["month"]=pd.to_datetime(wea_data['Timestamp']).dt.month


# In[47]:


dailymean1=pd.DataFrame(wea_data.groupby('date',as_index=False).agg({'demand':'mean','Temp.(F)':'mean','Humid.(%)':'mean'}))
dailymean1["month"]=pd.to_datetime(dailymean1['date']).dt.month
dailymean1.head()


# GRAPH 6

# In[55]:


#scatter-temp vs demand
scatter1=px.scatter(dailymean1, x='Temp.(F)',y='demand',title='Demand vs Temperature')
plotly_pane5 = pn.pane.Plotly(scatter1)
plotly_pane5


# GRAPH 7

# In[56]:


#scatter-humidity vs demand
scatter2=px.scatter(dailymean1, x='Humid.(%)',y='demand', title='Demand vs Humidity')
plotly_pane6 = pn.pane.Plotly(scatter2)
plotly_pane6


# GRAPH 8

# In[50]:


#heatmap-
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
corr = dailymean1.iloc[:,:-1].corr()
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns,annot=True)


# In[51]:


dailymean2=pd.DataFrame(wea_data.groupby('date',as_index=False).agg({'demand':'mean','Temp.(F)':'max','Humid.(%)':'mean'}))


# In[52]:


dailymean2.head()


# GRAPH 9

# In[53]:


corr = dailymean2.corr()
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns,annot=True)


# In[ ]:





# ##### building UI 

# In[317]:


#image=pn.Column(pn.pane.Str(background='#3f3f3f', sizing_mode='scale_both'), height=200)
image=pn.pane.JPG("C:/Users/HP/Desktop/teslaark.jpg",width=352,height=353)
img2=pn.pane.PNG("C:/Users/HP/Desktop/heatmap.png")
img3=pn.pane.PNG("C:/Users/HP/Desktop/capture.png",width=900)

img4=pn.pane.JPG("C:/Users/HP/Desktop/visu.jpg",width=1400,height=400)


# In[318]:


text="<font color ='red',font size =4>\n#DATA VISUALIZATION"


# In[319]:


#plot linking
p1 =('Peak demand-Histogram',pn.Row(pn.Column(plotly_pane0)))
tabs1 = pn.Tabs(p1)
tabs1.append(('Average demand-Histogram ',plotly_pane1))
tabs1.extend([('Daily Load curve',img3),('Monthly Load curve',plotly_pane3)])
a=tabs1
p2 =('Scatter plot 1',pn.Row(pn.Column(plotly_pane5)))
tabs2 = pn.Tabs(p2)
tabs2.append(('Scatter plot 2',plotly_pane6))
tabs2.extend([('Heatmap',img2)])
b=tabs2


# In[320]:


css = '''
.widget-box {
  background: #f0f0f0;
  border-radius: 5px;
  border: 1px black solid;
}
'''
pn.extension(raw_css=[css])


# In[321]:


text1 = "<font color ='white',font size =3>\n##DEMAND ANALYSIS"
#select=pn.widgets.Select(name='Select', options=['daily mean demand', 'monthly demand','decomposition plot'])
uni = pn.Column(pn.widgets.Toggle(name='UNIVARIATE', button_type='success'),a,
        css_classes=['widget-box'])

multi = pn.Column(pn.widgets.Toggle(name='MULTIVARIATE', button_type='danger'),b,
        css_classes=['widget-box'] )
p=pn.Row(pn.Row(uni,multi),background='black',width=2000,height=1000)


# In[322]:


cam1 = pn.Column(pn.Column(img4),text1,p,background='black',width=2000,height=1000)
cam1.show()


# In[ ]:




