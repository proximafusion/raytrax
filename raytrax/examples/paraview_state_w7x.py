# state file generated using paraview version 6.0.0
import paraview
paraview.compatibility.major = 6
paraview.compatibility.minor = 0

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.Set(
    ViewSize=[1098, 897],
    CenterOfRotation=[4.992373585700989, 1.8281000852584839, 0.032098591327667236],
    CameraPosition=[2.930945081972503, -5.316621505045068, -1.3314778381605725],
    CameraFocalPoint=[4.9923735857009826, 1.828100085258465, 0.03209859132766716],
    CameraViewUp=[-0.16332861990113573, 0.23021441375119708, -0.9593357522902933],
    CameraFocalDisk=1.0,
    CameraParallelScale=2.367620212141836,
    OSPRayMaterialLibrary=materialLibrary1,
)

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1098, 897)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Structured Grid Reader'
w7xvts = XMLStructuredGridReader(registrationName='w7x.vts', FileName=['w7x.vts'])
w7xvts.Set(
    PointArrayStatus=['B', 'absB', 'rho'],
    TimeArray='None',
)

# create a new 'Contour'
contour2 = Contour(registrationName='Contour2', Input=w7xvts)
contour2.Set(
    ContourBy=['POINTS', 'rho'],
    ComputeNormals=0,
    Isosurfaces=[0.8],
)

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=w7xvts)
contour1.Set(
    ContourBy=['POINTS', 'B_Magnitude'],
    ComputeNormals=0,
    Isosurfaces=[2.5],
)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from contour1
contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'rho'
rhoLUT = GetColorTransferFunction('rho')
rhoLUT.Set(
    RGBPoints=GenerateRGBPoints(
        range_min=0.3425925672054291,
        range_max=1.0,
    ),
    NanOpacity=0.0,
    ScalarRangeInitialized=1.0,
)

# trace defaults for the display properties.
contour1Display.Set(
    Representation='Surface',
    ColorArrayName=['POINTS', 'rho'],
    LookupTable=rhoLUT,
    SelectNormalArray='Normals',
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
contour1Display.ScaleTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
contour1Display.OpacityTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

# show data from contour2
contour2Display = Show(contour2, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'absB'
absBLUT = GetColorTransferFunction('absB')
absBLUT.Set(
    RGBPoints=GenerateRGBPoints(
        range_min=1.9251153469085693,
        range_max=2.818242073059082,
    ),
    NanOpacity=0.14,
    ScalarRangeInitialized=1.0,
)

# trace defaults for the display properties.
contour2Display.Set(
    Representation='Surface',
    ColorArrayName=['POINTS', 'absB'],
    LookupTable=absBLUT,
    SelectNormalArray='Normals',
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
contour2Display.ScaleTransferFunction.Points = [2.371678590774536, 0.0, 0.5, 0.0, 2.372166872024536, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
contour2Display.OpacityTransferFunction.Points = [2.371678590774536, 0.0, 0.5, 0.0, 2.372166872024536, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for absBLUT in view renderView1
absBLUTColorBar = GetScalarBar(absBLUT, renderView1)
absBLUTColorBar.Set(
    WindowLocation='Any Location',
    Position=[0.8852459016393442, 0.06688963210702342],
    Title='absB',
    ComponentTitle='',
)

# set color bar visibility
absBLUTColorBar.Visibility = 1

# get color legend/bar for rhoLUT in view renderView1
rhoLUTColorBar = GetScalarBar(rhoLUT, renderView1)
rhoLUTColorBar.Set(
    Title='rho',
    ComponentTitle='',
)

# set color bar visibility
rhoLUTColorBar.Visibility = 1

# show color legend
contour1Display.SetScalarBarVisibility(renderView1, True)

# show color legend
contour2Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'absB'
absBPWF = GetOpacityTransferFunction('absB')
absBPWF.Set(
    Points=[1.9251153469085693, 0.0, 0.5, 0.0, 2.818242073059082, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

# get opacity transfer function/opacity map for 'rho'
rhoPWF = GetOpacityTransferFunction('rho')
rhoPWF.Set(
    Points=[0.3425925672054291, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper

# get time animation track
timeAnimationCue1 = GetTimeTrack()

# initialize the animation track

# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.Set(
    ViewModules=renderView1,
    Cues=timeAnimationCue1,
    AnimationTime=0.0,
)

# initialize the animation scene

# ----------------------------------------------------------------
# restore active source
SetActiveSource(contour2)
# ----------------------------------------------------------------


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://www.paraview.org/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------