'''
Polarization Analysis Application
=================================



'''

import os
from threading import Timer
import webbrowser
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from emlib.quant.quants import Vector, Angle, Length, ObsPoints, Voltage
from emlib.fields.polarization import Polarization
#from pyquaternion import Quaternion
from dash import Dash, html, dcc, callback, Output, Input, State, dependencies
import dash_bootstrap_components as dbc
import dash_daq as daq
from _plotly_utils.colors.cmocean import amp
    





class PoincarePlot(go.FigureWidget):
    def __init__(self, dpols, amps, phases, effs, sphereMode='lightgrey'):
        '''
        pols is an array or list of Polarization objects to plot as a marker
        eff is an array or list of efficientcy  disks to plot about each pol, values are in percent
        '''
        self._pols = dpols
        self._effs = effs
        self._amp = amps
        self._phase = phases
        self._rng = np.random.default_rng() 
        self._sphMode = sphereMode  # must be color, or 'Power'
        self._lhE = []
        self._rhE = []
        
    
        
        # Create the polarizations resulting from driving the two dpols with
        # all the different combinations of amp and phase (NO random errors added)
        
        self._createPols()
        # print("_________________________")
        # for apol in self._pols:
            # print(apol)
        
        
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
                     zaxis = axisGlobal ), uirevision=self._rng.uniform(1, 100)) 
        
     
        #data holds all of the 3D objects (lines, sphere, markers, disks, erc)to be plotted
        # annotations are text place in 3D (polarizations of axes)
        data = []
        annotations = []
        
        # Create axis lines and labels
        labels = [['VP', 'HP'], ['Slant -45', 'Slant +45'], ['RHCP', 'LHCP']]
    
        for axis in range(3):
            line, axlabels = self._createAxis(axis=axis, labels=labels[axis])
            data.append(line)
            annotations = annotations + axlabels
            
       
        # This is the Poincare sphere reference.  A transparent sphere unless self._sphMode == 'Power'
        v = ObsPoints.makeUniformSphericalGrid(numPts=1000, radius=1.0, units=None)
        tri = ConvexHull(v)
        x = v[:,0]
        y = v[:,1]
        z = v[:,2]
        i = tri.simplices[:,0]
        j= tri.simplices[:,1]
        k= tri.simplices[:,2]

        if self._sphMode != 'Power':
            sph =  go.Mesh3d(x=x, y=y, z=z,
                            i=i, j=j, k=k, 
                            color=self._sphMode, 
                            opacity=0.2,
                            flatshading=True)
        else:
            #TODO: write PoimcarePlot method to generate power intensity from V (sph verticies)
            # d1v = dpols[0].poincareXyz()
            # d2v = dpols[1].poincareXyz()
        
            intensity = []

            for indx in range(int(v.size/3)):
                ampint = Polarization.dualPolPwrIntensity(dpols[0], dpols[1], Vector(v[indx,:]))
               
                intensity.append( ampint)
                # dot2 = np.dot(d2v[0,:], v[indx,:])
                # eta1 = np.arccos(dot1)
                # eta2 = np.arccos(dot2)
                #
                # angdelta = (eta1 + eta2)/np.pi
                #
                # if dot2 < dot1:
                #     zeta = np.arccos(dot1)/(2*angdelta)
                #     e1 = np.cos(zeta)
                #     e2 = np.sin(zeta)
                #     atten = e2 / e1
                #
                # else:
                #     zeta = np.arccos(dot2)/(2*angdelta)
                #     e1 = np.sin(zeta)
                #     e2 = np.cos(zeta)
                #     atten = e1 / e2
                #
                #
                #
                # atten = (1 + atten**2)/2
                # intensity.append(10*np.log10(atten))
              
            sph =  go.Mesh3d(x=x, y=y, z=z,
                            i=i, j=j, k=k, 
                            intensity=intensity,
                            intensitymode = 'vertex',
                            colorscale='Jet', 
                            opacity=1,
                            flatshading=True,
                             colorbar=dict(lenmode='fraction', len=0.75, thickness=20) )                
        
        data.append(sph)
        
        # save the current length of data.
        # the next items added will be the polarization markers and efficiency disks
        # this index location will be used for random variation animation
        self._polsIndx = len(data)
        #print(data)
        
        # Create graphical markers and efficiency disks for each pol
        # and add them to data.  the first four items added are for
        # the two drive polarizations (marker, effDisk, marker, effDisk, ...)
        for apol, aeff in zip(self._pols, self._effs):
            xyz = apol.poincareXyz()
            data.append(self._createPolMarker(xyz, color='blue'))
            xyzeff = self._getPolEffdiskXyz(apol, aeff)
            data.append(self._createPolEffdisk(xyzeff))
        
        super().__init__( data=data, layout=layout) #, frames=None, skip_invalid=False, **kwargs) 

    
        self.update_layout( scene=dict(    annotations=annotations ), )
        
    def _createPols(self):
        # self._pols contains the two initial drive polarizations.
        # using the amplitudes and phases in self._amp and self._phase
        # this method appends to self._pols all the resulting polarizations
        # this is run when submit is pushed
        #TODO: does not include random variations, but should it for submit?
        self._lhE = []
        self._rhE = []

        for anamp in self._amp:
            for aphase in self._phase:
                relAmp = 10**(anamp/20)
                polGen, lhE, rhE = Polarization.dualPolGeneration(self._pols[0], self._pols[1], Voltage(relAmp), Angle(aphase, 'deg'))
                self._pols.append(polGen)
                self._lhE.append(lhE) 
                self._rhE.append(rhE)
                print(f"LH E = {np.abs(lhE)}, RH E = {np.abs(rhE)} = {np.abs(lhE)**2 + np.abs(rhE)**2}")
        

    def _createPolsWithVariations(self, ramp, rphase):
        pols = []
        self._lhE = []
        self._rhE = []

        for anamp in self._amp:
            for aphase in self._phase:
                relAmp = 10**((anamp + self._rng.uniform(-ramp, ramp))/20)
                ranPhase = aphase + self._rng.uniform(-rphase, rphase)
                polGen, lhE, rhE = Polarization.dualPolGeneration(self._pols[0], self._pols[1], Voltage(relAmp), Angle(ranPhase, 'deg'))
                pols.append(polGen)
                #TODO:  Need to rethink how lhe and rhE during variation runs will work. 
                self._lhE.append(lhE) 
                self._rhE.append(rhE)
        
        return pols

    def _createAxis(self, axis=0, labels=['neg', 'pos']):
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


    
    def _createPolMarker(self, xyz, color='blue'):
        return  go.Scatter3d(x=xyz[:,0], 
                            y=xyz[:,1], 
                            z=xyz[:,2], 
                            mode='markers',
                            marker=dict(
                                        size=8,
                                        color=color,                # set color to an array/list of desired values
                                        colorscale='Viridis',   # choose a colorscale
                                        opacity=1.0
        ))
        
        
    def _getPolEffdiskXyz(self, pol, eff):
        '''
        pol is the polarization object
        eff is the  mismatch factor in percent
        eff = 100 * cos**2(ang/2)   (when diametrically opposite (180) = 0
        '''
        eang = 2 * np.arccos(np.sqrt(eff / 100.0))
        rf = 1.0
        
        phi = 2 * pol.tau 
        theta =Angle(90, units='deg') - 2 * pol.lamda
        r = Length([1])
        pvec = ObsPoints.fromSpherical( phi, theta, r, makeUnit=False) 
        edgevec = ObsPoints.fromSpherical( phi, theta+eang, r, makeUnit=False) 
        angInc = Angle(30, units='deg')
        qinc = Quaternion(axis=pvec.rv, radians=angInc)
        def cumRotate(qinc, startVec, angleStart, angleStop, angInc):
                #print(startvec)
        
                aVec = startVec
                #print(angleStart.u('deg'))
                #print(angInc)
                ang = angleStart
                minAngDiff = Angle(3*np.pi)
                curAngDiff = Angle(2*np.pi)
                while curAngDiff < minAngDiff:
                    minAngDiff = curAngDiff
                    yield aVec
                    ang += angInc
                    
                    curAngDiff = Angle(np.abs(angleStop - ang)) #.normPMpi()
                    
                    #print(f"{ang.u('deg')} d = {curAngDiff.u('deg')}")
                    aVec = Vector(qinc.rotate(aVec.rv))
                #print("ignore last")
        meshpts = ObsPoints([v for v in cumRotate(qinc, edgevec, Angle(0 ,units='deg'), Angle(330,units='deg'), angInc)], units='unit')
        meshpts = np.concatenate((meshpts, pvec), axis=0)
        #tri = ConvexHull(meshpts)
        return rf*meshpts
    
    def _createPolEffdisk(self, xyz):    
        i = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
        j = [1,   2,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]
        k = [0,   1,  3,  4,  5,  6,  7,  8,  9, 10, 11,  0]
        cmesh = go.Mesh3d(x=xyz[:,0], 
                          y=xyz[:,1], 
                          z=xyz[:,2], 
                          i=i, j=j, k=k, 
                          color='green',
                          flatshading= True,
                          opacity=0.5)     
        
        return cmesh 
    
    def _updateGOcoords(self,fig,  dataIndx, xyz):
        fig['data'][dataIndx]['x'] = xyz[:,0]
        fig['data'][dataIndx]['y'] = xyz[:,1]
        fig['data'][dataIndx]['z'] = xyz[:,2]
              
        
    def runRandom(self, fig,  ramp, rphase):
        
        #fig['layout']['uirevision'] = self._rng.uniform(1, 100)
        # _polsIndx is the location of the first drive pol's
        # marker.  Add 4 to get location of first generated
        # pol marker
        dataIndx = self._polsIndx + 4
        
        # newPols does not include the two drive pols
        newPols = self._createPolsWithVariations(ramp, rphase)
        for apol, aneff in zip(newPols, self._effs[2:]):
            # get new marker positions and update
            xyz = apol.poincareXyz()
            self._updateGOcoords(fig, dataIndx, xyz)
            
            # get new eff disk positions and update 
            dataIndx += 1
            xyz = self._getPolEffdiskXyz(apol, aneff )
            self._updateGOcoords(fig, dataIndx, xyz)
            dataIndx += 1

        return fig   


def create_card(card_id, title, description, initVal, ctype='number'):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4(title, id=f"{card_id}-title"),
                dcc.Input( id=card_id,
                            placeholder='Enter a value...',
                            type=ctype,
                            value=initVal,
                            style={'width': '80%', 'height':'20%'} ),
                html.P(description, id=f"{card_id}-description")
            ],
            style= dict(
               paddingLeft = 10,
               paddingRight = 0,
               paddingTop = 0,
               paddingBottom =0
                )
        ),
     style= dict(
               paddingLeft = 10, 
               paddingRight = 0,
               paddingTop = 0,
               paddingBottom =0,
               width="100%",
               height='80%'
                )   
    ) 
    
    
    
 
    
class PolarizationApp():
    def __init__(self):   
        self.app = Dash(external_stylesheets=[dbc.themes.MORPH], 
                        meta_tags=[
                            {"name": "viewport", "content": "width=device-width, initial-scale=1"},
                            ],prevent_initial_callbacks=True)
    
        
        self.app.layout = html.Div([
            html.Div([
                html.H1(children='Polarization App', style={'textAlign':'center'}),
                dbc.Row([
                        dbc.Col(html.H2("Drive 1"), width=3),
                        dbc.Col(html.H2("Excitations"))
                        ]),
                dbc.Row([
                    dbc.Col([create_card('d1tau', 'tau', 'degree', 0)], width=2), 
                    dbc.Col([create_card('d1lamda', 'lamda', 'degree', 0)], width=2, style= dict( paddingLeft = 0)),
                    dbc.Col([create_card('amptxt', 'Amplitude', 'in dB', '0', ctype='text')], width=4),
                    dbc.Col([create_card('phasetxt', 'Phase', 'in degree', '0', ctype='text')], width=4, style= dict( paddingLeft = 0) ),
                    ]),
                dbc.Row([
                    dbc.Col(html.H2("Drive 2"), width=3),
                    dbc.Col(html.H2('Polarization Efficiency'))
                    ]),
                dbc.Row([ 
                    dbc.Col([create_card('d2tau', 'tau', 'degree', 90)], width=2), 
                    dbc.Col([create_card('d2lamda', 'lamda', 'degree', 0)], width=2, style= dict( paddingLeft = 0)),
                    dbc.Col([create_card('efftxt', 'Efficiency', 'percentage','99', ctype='text')], width=4),
                    dbc.Col([html.H2("Random Variation")], width=3),
                    ]),
                dbc.Row([
                    dbc.Col(html.Button('Submit', id='submit-button')),
                    dbc.Col([dcc.Interval( id="randomtrig", interval=1 * 300, n_intervals=0 , disabled=True)], width=3),
                    dbc.Col([create_card('amprandom', 'Amp', 'in dB', 0)], width=2),
                    dbc.Col([create_card('phaserandom', 'Phase', 'in degrees', 0)], width=2, style= dict( paddingLeft = 0) )
                    ]),
                dbc.Row([
                    dbc.Col(['Enter values and press submit'], width=3),
                    dbc.Col([daq.PowerButton( id="start", 
                                              on=False, 
                                              color='green', 
                                              size=80,
                                              label='Run Random Variations',
                                              labelPosition='top')]),
                    dbc.Col([dcc.Dropdown(['lightgrey', 'Power', 'red'], 'lightgrey', id='sphereColor'),])
                    ]),

            ], style={'padding': 10, 'flex': 1}),
        
            html.Div([    dcc.Graph(id='graph-content', style=dict(backgroundColor='black', width="60%"))],
              style={'padding': 10, 'flex': 1})
            
            ], style={'display': 'flex', 'flexDirection': 'row'})
        

        
        self.app.callback(
            dependencies.Output('graph-content', 'figure',allow_duplicate=True),
            [dependencies.Input("submit-button", "n_clicks"),
            dependencies.State('d1tau', 'value'),
            dependencies.State('d1lamda', 'value'),
            dependencies.State('d2tau', 'value'),
            dependencies.State('d2lamda', 'value'),
            dependencies.State('amptxt', 'value'),
            dependencies.State('phasetxt', 'value'),
            dependencies.State('efftxt', 'value'),
            dependencies.State('sphereColor', 'value')],
            prevent_initial_call=True
            )(self.update_graph)
            
        self.app.callback( 
                            dependencies.Output('graph-content', 'figure'),
                            [ dependencies.Input("randomtrig", "n_intervals")],
                            [dependencies.State('graph-content', 'figure'),
                            dependencies.State('amprandom', 'value'),
                            dependencies.State('phaserandom', 'value')],
                            prevent_initial_call=True
                          )(self.randomRun)
                
        self.app.callback(
                            dependencies.Output('randomtrig', 'disabled'),
                            [dependencies.Input("start", 'on'),],
                            [State('randomtrig', 'disabled')]
                            ) (self.callback_func_start_stop_interval)

    def callback_func_start_stop_interval(self, starton, disabled_state):
        if starton:
            return False
        else:
            return True   
            
            
    def update_graph(self, n_clicks, d1tau, d1lamda, d2tau, d2lamda, amptxt, phasetxt, efftxt, sphereColor):
        #print("Update graph called")
        dpols = [Polarization(Angle(d1tau, 'deg'), Angle(d1lamda, 'deg')),
                Polarization(Angle(d2tau, 'deg'), Angle(d2lamda, 'deg'))]
        amps = eval('['+amptxt+']')
        phases = eval('['+phasetxt+']')
        effs =  eval('['+efftxt+']')
        if len(effs) == 0:
            effs = [100]
            
        numOfPols = 2 + len(amps)*len(phases)
        if len(effs) < numOfPols:
            effs = effs + [effs[-1] for i in range(numOfPols - len(effs))]
            
        self.pPlot = PoincarePlot(dpols, amps, phases, effs, sphereMode=sphereColor)
        return self.pPlot


    def randomRun(self, n, fig, ramp, rphase):
        #print("RandomRun called")
        if fig is not None:
            
            self.pPlot.runRandom(fig, ramp, rphase)
        return fig

    def run(self,debug=True, port=6000):
        self.app.run(debug=debug, host='127.0.0.1', port=port)
    
def open_browser():
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:1222/')

if __name__ == "__main__":
    app = PolarizationApp()
    Timer(1, open_browser).start()
    app.run(debug=True, port=1222)    
  
    
    
