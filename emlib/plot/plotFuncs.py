'''
Plotting Functions
~~~~~~~~~~~~~~~~~~

'''
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from ..quant.quants import Angle, Vector, Quaternion, ObsPoints
from ..antArrays.antArray import AntArray
from ..antennas.antBaseClass import AntBase
from ..fields.emField import EMField, ArrayFactor

def polarPlot(pat, refPol=None, range_r=None):
    '''
    range_r (list of two numbers)
    '''
    if isinstance(pat,  ArrayFactor):
        amp = pat.volts.u('dB')
    elif isinstance(pat,  EMField):
        amp = pat.responseTo(refPol).u('dB')
    else:
        raise Exception('polarPlot only takes an EMField or ArrayFactor object')

    if range_r is not None:
        amp = np.clip(amp, a_min=range_r, a_max=None)

    
    if pat.obsPts.pattype == "Azimuth":
        pangle = -pat.obsPts.phi 
        angdir = "clockwise"
        plotRot = 90
        ticktext = np.concatenate((10 * ( np.arange(19)), np.flip(-10*np.arange(18))))
        titletext=f"Azimuth plot at el = {pat.obsPts.patinfo['Fixed'].u('deg'):.1f}\u00B0"
            
    elif pat.obsPts.pattype == "Phi":
        pangle = pat.obsPts.phi 
        angdir = "counterclockwise"
        plotRot = 90
        ticktext =  np.concatenate((10 * ( np.arange(19)), np.flip(-10*np.arange(18))))
        titletext=f"Phi plot at Theta = {pat.obsPts.patinfo['Fixed'].u('deg'):.1f}\u00B0"
            
    elif pat.obsPts.pattype == "Elevation":
        mask = (False == np.isclose(pat.obsPts.patinfo['Fixed'].u(), -pat.obsPts.phi.u()))
        pangle = mask*Angle(np.pi) + (Angle(np.pi/2) - pat.obsPts.theta)*(mask*-2 + 1) 
        angdir = "counterclockwise"
        plotRot = 0
        ticktext = np.concatenate((10*np.arange(10), 10*np.flip(np.arange(9)), -10*(1 + np.arange(9)), -10*np.flip(1 + np.arange(8))))
        titletext=f"Elevation plot at Az = {pat.obsPts.patinfo['Fixed'].u('deg'):.1f}\u00B0"
          
    elif pat.obsPts.pattype == "Theta":
        mask = (False == np.isclose(pat.obsPts.patinfo['Fixed'].u(), pat.obsPts.phi.u()))
        pangle =  (pat.obsPts.theta)*(mask*-2 + 1) 
        angdir = "clockwise"
        plotRot = 90
        ticktext = np.concatenate((10*np.arange(19), 10*np.flip(1 + np.arange(17))))
        titletext=f"Theta plot at Phi = {pat.obsPts.patinfo['Fixed'].u('deg'):.1f}\u00B0"
            
    elif pat.obsPts.pattype == "Arbitrary":
        pangle = pat.obsPts.phi  #requires more work
        angdir = "counterclockwise"
        plotRot = -90
            
    else:
        raise Exception("Unknown patType")            
         

    ticktext = ["%.0f" % number for number in ticktext]
    #print(ticktext)
    #print(["%.0f"% num for num in pangle.u('deg')])
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r = amp,
        thetaunit='radians',
        theta = pangle,
        mode = 'lines',
        name = 'Figure 8',
        line_color = 'peru'
    ))
    fig.update_layout(
        title=dict(text=titletext),
        autosize=False,
        width=1000,
        height=1000,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        paper_bgcolor="white",
        polar = dict(
            hole=0.02,
            bgcolor='white',
            angularaxis = dict(
                tickfont_size=16,
                rotation=plotRot, # start position of angular axis
                direction=angdir,
                gridcolor="grey",
                griddash='dot',
                linecolor='black',
                tickmode='array',
                tickvals= 10*np.arange(36),
                ticktext=ticktext
                ),
            radialaxis = dict(title=dict(text='dB'),
                              griddash='dot',
                              tickfont_size=16,
                              gridcolor='grey',
                              linecolor='black')
    ))
    fig.show()

def createAxis( axis=0, labels=['neg', 'pos']):
       r = 1.1
       xyz = np.array([[0,0],[0,0],[0,0]])
       xyz[axis,:] = [-r, r]
       lines = go.Scatter3d(x=xyz[0, :], 
                           y=xyz[1, :], 
                           z=xyz[2, :], 
                           mode='lines', 
                           line = dict(width=2, color='black'))
       labels = [
                   dict(
                       showarrow=False,
                       x=xyz[0, 0],
                       y=xyz[1, 0],
                       z=xyz[2, 0],
                       text=labels[0],
                       xanchor="left",
                       yanchor="bottom",
                       opacity=1),
                   dict(
                       showarrow=False,
                       x=xyz[0, 1],
                       y=xyz[1, 1],
                       z=xyz[2, 1],
                       text=labels[1],
                       xanchor="left",
                       yanchor="bottom",
                       opacity=1),]
       return (lines, labels)
   
    
def drawEclipse(pol, obsPtxyz, scale):
    r = pol.axialRatio
    
    if np.sign(r) < 0:
        lcolor = 'red'
    else:
        lcolor = 'blue'
    
    r = np.abs(r)
    #print(f"{r}. tau={pol.tau.u('deg')} lamda = {pol.lamda.u('deg')} at theta = {obsPtxyz.theta.u('deg')}, phi = {obsPtxyz.phi.u('deg')}")
    if r>100:
        x = [0, 0]
        y = [-scale, scale]
        z = [1, 1]
    elif r < 1:
        x = [scale*r*np.cos(a*30*np.pi/180) for a in np.arange(13)]
        y = [scale*r*np.sin(a*30*np.pi/180) for a in np.arange(13)] 
        z = [ 1 for a  in np.arange(13)]
    else:
        x = [scale*r*np.cos(a*30*np.pi/180) for a in np.arange(13)]
        y = [scale*r*np.sin(a*30*np.pi/180) for a in np.arange(13)]
        z = [ 1 for a  in np.arange(13)]
        
    exyzs = Vector(np.swapaxes([x, y, z],0,1))
    
    
    # quaternion to rotate major axis to be perpendicuar to obsPtxyz phi angle 
    q1 = Quaternion(Angle(obsPtxyz.phi), Vector([0, 0, 1]))
    
    # find perpendicular to both z and obsPtxyz
    rotaxis = np.cross([0, 0, 1], obsPtxyz)
    
    # find angle between z and obsPtxyz
    rotang = np.arccos(np.dot([0, 0, 1], obsPtxyz[0,:]))
    
    # quaternion to rotate center of elipse to obsPtxyz
    q2 = Quaternion(Angle(rotang), Vector(rotaxis).norm)
    
    # quaternion to rotate to tilt angle
    q3 = Quaternion(pol.tau, obsPtxyz.norm)

    
    qt = q3*q2*q1
    exyzs = qt.forwardRotate(exyzs)
    
    # create go.scatter3d object
    return go.Scatter3d(
                        x=exyzs[:,0],
                        y=exyzs[:,1],
                        z=exyzs[:,2],
                        mode = 'lines', 
                        line = dict(color = lcolor) )

    
def Plot3Dpol(pat, scale): 
    data = []
    annotations = []
    for indx, xyz in enumerate(pat.obsPts):
        data.append(drawEclipse(pat.pol[indx], Vector(xyz), scale))       
    
    
    #sphere reference
    v = ObsPoints.makeUniformSphericalGrid(numPts=1000, radius=0.99, units=None)
    tri = ConvexHull(v)
    x = v[:,0]
    y = v[:,1]
    z = v[:,2]
    i = tri.simplices[:,0]
    j= tri.simplices[:,1]
    k= tri.simplices[:,2]
    sph =  go.Mesh3d(x=x, y=y, z=z,
                i=i, j=j, k=k, 
                color='white', 
                opacity=1.0,
                flatshading=True)
    data.append(sph)
    
    labels = [['-X', '+X'], ['-Y', '+Y'], ['-Z', '+Z']]

    for axis in range(3):
        line, axlabels = createAxis(axis=axis, labels=labels[axis])
        data.append(line)
        annotations = annotations + axlabels
    
    axisGlobal = dict(showgrid = False,
                      showticklabels = False,
                      showbackground = False,
                      showline = False,
                      showaxeslabels = False,
                      title={'text':''} )    
        
    layout = go.Layout(
    autosize=False,
    showlegend=False,
    width=1000,
    height=1000,
    margin_b=1, margin_l=1, margin_t=1, margin_r=1,
    xaxis=go.layout.XAxis(linecolor="black", linewidth=1, mirror=True),
    yaxis=go.layout.YAxis(linecolor="black", linewidth=1, mirror=True),
    margin=go.layout.Margin(l=50, r=50, b=100, t=100, pad=4),
    scene = dict(xaxis = axisGlobal,
                 yaxis = axisGlobal,
                 zaxis = axisGlobal,
                 annotations=annotations ))
    fig = go.Figure( data=data, layout=layout)
    fig.show()   
 
    
    
def Plot3Dpat(pat, refPol=None, rangeMin_r=None):
    '''
    Plots EMfield in 3D by generating mesh from convex hull triangulation
    '''
    tri = ConvexHull(pat.obsPts)
    if isinstance(pat,  ArrayFactor):
        amp = pat.volts.u('dB')
    elif isinstance(pat,  EMField):
        amp = pat.responseTo(refPol).u('dB')
    else:
        raise Exception('Plot3Dpat only takes an EMField or ArrayFactor object')
    
    if rangeMin_r is not None:
        ampc = np.clip(amp, a_min=rangeMin_r, a_max=None)
    else:
        ampc = amp
        
    colorVal = ampc 
    ampc = ampc - np.min(ampc)
    thetas = pat.obsPts.theta.u('deg')
    phis = pat.obsPts.phi.u('deg')
    hovertext = [f"\u03B8={thetas[i]:.1f}\u00B0, \u03D5={phis[i]:.1f}\u00B0, {colorVal[i]:.1f} dB"  for i in range(colorVal.size)]
    

   # amp = 1
    x = ampc*pat.obsPts[:,0]
    y = ampc*pat.obsPts[:,1]
    z = ampc*pat.obsPts[:,2]
    
    i = tri.simplices[:,0]
    j= tri.simplices[:,1]
    k= tri.simplices[:,2]
    
    axisGlobal = dict(showgrid = False,
                      showticklabels = False,
                      showbackground = False,
                      showline = False )
    
    layout = go.Layout(
    autosize=False,
    width=1000,
    height=1000,
    xaxis=go.layout.XAxis(linecolor="black", linewidth=1, mirror=True),
    yaxis=go.layout.YAxis(linecolor="black", linewidth=1, mirror=True),
    margin=go.layout.Margin(l=50, r=50, b=100, t=100, pad=4),
    scene = dict(xaxis = axisGlobal,
                 yaxis = axisGlobal,
                 zaxis = axisGlobal ))    
    
    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,
                                    intensity = colorVal, 
                                    intensitymode = 'vertex',
                                    i=i, j=j, k=k, 
                                    colorscale='Jet',
                                    hoverinfo='text',
                                    hovertext=hovertext, 
                                    opacity=1.0)],
                                    layout=layout)
    

    fig.show()
    

def Plot3D(objs):
    if not isinstance(objs, list):
        objs = [objs]
    data = []
    mline = np.array([[0, 0, 0],
                     [0, 0, 0]])
    first = True
    for obj in objs:
        if isinstance(obj, AntBase):
            vert, tri = obj.get3DModel()
            data.append( go.Mesh3d(x=vert[:,0], y=vert[:,1], z=vert[:,2],
                                    i=tri[:,0], j=tri[:,1], k=tri[:,2], 
                                    color='brown', 
                                    opacity=1.0))
            
        if isinstance(obj, AntArray):

            data.append(go.Scatter3d(x=obj[:,0], y=obj[:,1], z=obj[:,2], mode='markers'))
            
            
   
            
            

    #data.append(go.Scatter3d(x=mline[:,0], y=mline[:,1], z=mline[:,2], mode='lines'))                
        
    layout = go.Layout(
        scene=dict(
                 aspectmode='data'
         ),
    autosize=True,
    xaxis=go.layout.XAxis(linecolor="black", linewidth=1, mirror=True),
    yaxis=go.layout.YAxis(linecolor="black", linewidth=1, mirror=True),
    margin=go.layout.Margin(l=50, r=50, b=100, t=100, pad=4))    
    
    
    
    
    fig = go.Figure(data=data, layout=layout)
     
        
    fig.show()



    