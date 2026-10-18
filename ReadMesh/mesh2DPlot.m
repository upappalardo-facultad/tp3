function plotHandle = mesh2DPlot(elements, nodes, options)
% Mesh plotter
% 
% mesh2DPlot(elements, nodes, options)
% 
% Nodal order in elementNodesArray:
%  Q4 Q8 Q9        CST LST
%
%  4---7---3        3
%  |       |        | \              
%  8   9   6        6  5   
%  |       |        |    \     
%  1---5---2        1--4--2
% inputs are nodes and elements structures  
% nodes needs to have following variables:
% id (nNod x 1 array)
% position (nNod x 2/3 array, but only uses x and y coordinates)
% element needs to have:
% id
% connectivity
% supports mixed elements
% options allow for graphic plotting options changing

arguments
    elements struct
    nodes struct
    options.NodeLabel (1, 1) {mustBeNumericOrLogical} = true;
    options.ElementLabel  (1, 1) {mustBeNumericOrLogical} = true;
    options.ElementColor = 'k';
    options.NodeColor = 'r';
    options.FontSize (1, 1) {mustBeInteger} = 8;
    options.MarkerSize (1, 1) {mustBeInteger} = 10;
    options.DisplayName {mustBeText} = '';
    options.LineWidth {mustBeNumeric} = 2;
end


%% Definition
% Steering vector reorder definition
% define order for CST, LST, Q4, Q8, Q9 elements
nodeOrderTable = table([3 4 6 8 9]', {[1 2 3]; [1 2 3 4]; [1 4 2 5 3 6]; [1 5 2 6 3 7 4 8]; [1 5 2 6 3 7 4 8]}, 'VariableNames', {'nNodEle', 'nodeOrder'});

%% Object definition - define patches for all element types

nNodEles = unique(cellfun(@length, elements.connectivity));
nElementSize = length(nNodEles);

plotHandle = cell(nElementSize+1, 2);
for iElementSize = 1:nElementSize
    nNodEleIter = nNodEles(iElementSize);
    elementSizeFilter = cellfun(@(x) length(x) == nNodEleIter, elements.connectivity);
    elementNodeOrder = nodeOrderTable(ismember(nodeOrderTable.nNodEle, nNodEleIter), :).nodeOrder{1};
    elementSizeConnectivitiesCell = elements.connectivity(elementSizeFilter, :);
    elementSizeConnectivities = cat(1, elementSizeConnectivitiesCell{:});
    elementSizeIDs = elements.id(elementSizeFilter); 
    elementsSizeNodeIDs = elementSizeConnectivities(:, elementNodeOrder);
    [~, nodeIDarrayPosition] = ismember(elementsSizeNodeIDs, nodes.id);   
    plotHandle{iElementSize, 1} = patch('Faces', nodeIDarrayPosition,'Vertices', nodes.position, 'FaceColor', 'none');
    % for stress/strain/displacement contour plotting, investigate FaceVertexCData property
    set(plotHandle{iElementSize, 1},'EdgeColor',options.ElementColor, 'LineWidth', options.LineWidth);

    nEle = size(elementSizeConnectivities, 1);

    % element label optional
    if options.ElementLabel
        xCoordinateElementCenter = zeros(nEle, 1);
        yCoordinateElementCenter = zeros(nEle, 1);
        for iEle = 1:nEle
            xCoordinateElementCenter(iEle) = mean(nodes.position(nodeIDarrayPosition(iEle, :), 1));
            yCoordinateElementCenter(iEle) = mean(nodes.position(nodeIDarrayPosition(iEle, :), 2));
        end
         plotHandle{iElementSize, 2} = text(xCoordinateElementCenter,yCoordinateElementCenter,num2str(elementSizeIDs),'EdgeColor',options.ElementColor,'Color',options.ElementColor,'FontSize',options.FontSize);
    end

end
    
set(gca,'XTick',[],'YTick',[],'XColor',[1 1 1],'YColor',[1 1 1]);
daspect([1 1 1]); hold on;

%% Plot nodes

plotHandle{end, 1} = plot(nodes.position(:, 1), nodes.position(:, 2),'o','MarkerFaceColor','r','MarkerEdgeColor','b','MarkerSize',4);

% node label optional
if options.NodeLabel
    plotHandle{end, 2} = text(nodes.position(:, 1), nodes.position(:, 2), num2str(nodes.id), 'VerticalAlignment','bottom','Color',options.NodeColor,'FontSize',options.FontSize);
end

end