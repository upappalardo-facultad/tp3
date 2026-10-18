clear
close all
clc

meshFilePath = 'mesh_dat_files\stress_concentration_mixed.dat';
mesh = importMeshFromDatFile(meshFilePath);
mesh2DPlot(mesh.elements, mesh.nodes, FontSize=6)