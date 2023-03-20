import vtk
import numpy as np

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


#
# DeformationExtractor
#

class DeformationExtractor(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "DeformationExtractor"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Andy Huynh (University of Western Australia)", "Benjamin Zwick (Universiy of Western Australia)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#DeformationExtractor">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

#
# DeformationExtractorWidget
#

class DeformationExtractorWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/DeformationExtractor.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = DeformationExtractorLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputSurfaceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI) #vtkMRMLModelNode
        self.ui.inputTransformSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI) #vtkMRMLGridTransformNode
        self.ui.outputSurfaceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI) #vtkMRMLModelNode

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """ 
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)


    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    
        # Initial GUI update
        self.updateGUIFromParameterNode()
    
    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.inputSurfaceSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputSurface"))
        self.ui.inputTransformSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputTransform"))
        self.ui.outputSurfaceSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputSurface"))

        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputSurface", self.ui.inputSurfaceSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputTransform", self.ui.inputTransformSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputSurface", self.ui.outputSurfaceSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self) -> None:
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            self.logic.process(self.ui.inputSurfaceSelector.currentNode(), self.ui.inputTransformSelector.currentNode(), self.ui.outputSurfaceSelector.currentNode())

#
# DeformationExtractorLogic
#

class DeformationExtractorLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def getDisplacementVectors(self, inputSurface, deformedSurface):

        # Get points from models
        undeformed_points = inputSurface.GetPolyData().GetPoints()
        deformed_points = deformedSurface.GetPolyData().GetPoints()

        # Calculate displacement vectors
        num_points = deformed_points.GetNumberOfPoints()
        displacement_vectors = vtk.vtkDoubleArray()
        displacement_vectors.SetNumberOfComponents(3)
        displacement_vectors.SetNumberOfTuples(num_points)
        displacement_vectors.SetName('DisplacementVectors')

        for i in range(num_points):
            undeformed_point = np.array(undeformed_points.GetPoint(i))
            deformed_point = np.array(deformed_points.GetPoint(i))
            displacement_vector = deformed_point - undeformed_point
            displacement_vectors.SetTuple(i, displacement_vector)

        print(displacement_vectors.GetTuple(round(num_points/2)))
        return displacement_vectors

        

    def process(self,
                inputSurface,
                inputTransform,
                outputSurface) -> None:
        
        # Clone input surface to outputSurface
        # Create a new vtkPolyData and DeepCopy the source model's polydata
        #cloned_polydata = vtk.vtkPolyData()
        #cloned_polydata.DeepCopy(inputSurface.GetPolyData())
        #outputSurface.SetAndObservePolyData(cloned_polydata)

        outputSurface.CopyContent(inputSurface)
        #outputSurface.Modified()
        
        # Clone input surface and apply transform to produce deformed surface
        deformedSurface = slicer.vtkMRMLModelNode()
        deformedSurface.SetName('DeformedSurface')
        deformedSurface.CopyContent(inputSurface)
        deformedSurface.SetAndObserveTransformNodeID(inputTransform.GetID())
        deformedSurface.Modified()
        slicer.mrmlScene.AddNode(deformedSurface)

        displacementVectors = self.getDisplacementVectors(inputSurface, deformedSurface)

        # Add the displacement vectors as a point data attribute
        undeformed_polydata = outputSurface.GetPolyData()
        undeformed_polydata.GetPointData().SetScalars(displacementVectors)
        outputSurface.Modified()

        # print(inputSurface.GetPolyData())
        print(deformedSurface)
        # print(outputSurface)
        # print(undeformed_polydata)




