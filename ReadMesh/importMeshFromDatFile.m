
function mesh = importMeshFromDatFile(file)

% importMeshFromDatFile: NASTRAN ".dat" extension output mesh importer         
% -------------------------------------------------------------------------
% input: file path of ".dat" extension                                       
% output: mesh structure with fields nodes and elements                    
% -------------------------------------------------------------------------
% nodes structure fields only reads GRID cards                                                   
% id: nNod x 1 id double precision vector                                  
% position: nNod x 3 position double precision array (x, y, z)             
% -------------------------------------------------------------------------
% element read supports:
% solid elements without material orientation CHEXA, CTETRA,
% planar elements without mesh specific thickness 
% CPLSTS(3 4 6 8), CPLSTN(3 4 6 8)
% shell elements without mesh specific thickness 
% CQUAD4, CQUAD8, CTRIA6, CTRIA3
%
% element structure fields:
% id: nEle x 1 double precision vector with element ID number              
% type: nEle x 1 string vector with element type                           
% property: nEle x 1 double precision vector with element property number  
% connectivity: nEle x 1 cell array with 1 x nNodEle double                
%               precision vectors(cell to allow for elements with          
%               different amount of nodes)                                   
%

%% Get lines strings, and compress cards spanning multiple lines
fileID = fopen(file);
c = textscan(fileID, '%s', 'Delimiter', '\n', 'Whitespace', '');
fclose(fileID);
textLines = c{1};
lines2delete = startsWith(textLines,  {'*', '+'});
lines2replace = endsWith(textLines, '+');

leftText = cellfun(@(x) x(1:end-1), textLines(lines2replace),'UniformOutput', false);
rightText = cellfun(@(x) x(9:end), textLines(lines2delete),'UniformOutput', false);
newLines = join([leftText rightText], '');
textLines(lines2replace) = newLines;
textLines(lines2delete) = [];

textLines(startsWith(textLines, '$')) = []; % delete comments
textLines(cellfun(@(x) length(x)<=8, textLines)) = []; % delete comments

%% Supported cards
supportedNodes = {'GRID'};


supportedElements = {'CHEXA', 'CTETRA', ...
                    'CPLSTS3', 'CPLSTS4', 'CPLSTS6', 'CPLSTS8', ...
                    'CPLSTN3', 'CPLSTN4', 'CPLSTN6', 'CPLSTN8', ...
                    'CQUAD4', 'CQUAD8', 'CTRIA3', 'CTRIA6'};


% Expand to large field format
supportedNodes = [supportedNodes cellfun(@(x) [x '*'], supportedNodes, 'UniformOutput', false)];
supportedElements = [supportedElements cellfun(@(x) [x '*'], supportedElements, 'UniformOutput', false)];
supportedTokens = [supportedNodes, supportedElements];

allSupportedCards = textLines(contains(textLines, supportedTokens));

%% Process supported elements
elementCards = allSupportedCards(contains(allSupportedCards, supportedElements));

presentElementTypes = unique(strtrim(cellfun(@(x) x(1:8), elementCards, 'UniformOutput', false)));

for iPresentElements = 1:length(presentElementTypes)
    eleType2Process = presentElementTypes{iPresentElements};
    if iPresentElements == 1 
        mesh.elements = readElements(elementCards, eleType2Process);
    else
        elementsSubset = readElements(elementCards, eleType2Process);
        mesh.elements.id = [mesh.elements.id; elementsSubset.id];
        mesh.elements.type = [mesh.elements.type; elementsSubset.type];
        mesh.elements.property = [mesh.elements.property; elementsSubset.property];
        mesh.elements.connectivity = [mesh.elements.connectivity; elementsSubset.connectivity];
    end
end

%% Process supported nodes
nodeCards = allSupportedCards(contains(allSupportedCards, supportedNodes));
mesh.nodes = readNodes(nodeCards);


%% Element reading function
function elements = readElements(elementCards, eleType)

eleTypeCards = elementCards(contains(elementCards, eleType));
eleTypeCards = cellfun(@(x) x(9:end), eleTypeCards, 'UniformOutput', false);
isLargeFormat = contains(eleType, '*');
if isLargeFormat
    fieldSize = 16;
else
    fieldSize = 8;
end
eleType = strrep(eleType, '*', '');

nEle = size(eleTypeCards, 1);

switch eleType
    case {'CHEXA', 'CTETRA'}    
        id = zeros(nEle, 1);
        property = zeros(nEle, 1);
        connectivity = cell(nEle, 1);
        type = cell(nEle, 1);
        for iEle = 1:nEle
            iEleCard = eleTypeCards{iEle};
            nFields = length(iEleCard)/fieldSize;
            nNodEle = nFields - 2;
            data = cell2mat(textscan(iEleCard, repmat(['%' num2str(fieldSize) 'd'], 1, nFields)));
            id(iEle) = data(1);
            property(iEle) = data(2);
            connectivity{iEle} = data(3:end);
            type{iEle} = [eleType num2str(nNodEle)];
        end
    case {'CPLSTS3', 'CPLSTS4', 'CPLSTS6', 'CPLSTS8', ...
                    'CPLSTN3', 'CPLSTN4', 'CPLSTN6', 'CPLSTN8', ...
                    'CQUAD4', 'CQUAD8', 'CTRIA3', 'CTRIA6'}
        nNodEle = str2double(eleType(end));
        nFields = (nNodEle+2); % id, property, node connectivity, % remaining fields unsupported
        eleTypeCards = cellfun(@(x) x(1:(fieldSize*nFields)), eleTypeCards, 'UniformOutput', false);
        cellData = cellfun(@(x) textscan(x, repmat(['%' num2str(fieldSize) 'd'], 1, nFields)), eleTypeCards, 'UniformOutput', false);
        data = cell2mat(cat(1, cellData{:}));
        id = data(:, 1);
        property = data(:, 2);
        connectivity = mat2cell(data(:, 3:end), ones(nEle, 1), nNodEle);
        type = repmat({eleType}, nEle, 1);
end

elements.id = id;
elements.type = type;
elements.property = property;
elements.connectivity = connectivity;

end

%% Node reading function

function nodes = readNodes(nodeCards)
    isLargeFormat = cellfun(@(x) contains(x(1:8), '*'), nodeCards);
    shortFormatCards = nodeCards(~isLargeFormat);
    largeFormatCards = nodeCards(isLargeFormat);
    nFields = 6;
    startOffset = 8;

    if ~isempty(shortFormatCards)
        nSizeShort = 8;
        posShort = startOffset + [1+nSizeShort*(0:(nFields-1)); nSizeShort*(1:nFields)];
        nodeArrayShort = cell2mat(cellfun(@(x) str2double(arrayfun(@(y, z) x(y:z), posShort(1, :), posShort(2, :), 'UniformOutput', false)), shortFormatCards, 'UniformOutput', false));
    else
        nodeArrayShort = [];
    end

    if ~isempty(largeFormatCards)
        nSizeLarge = 16;
        posLarge = startOffset + [1+nSizeLarge*(0:(nFields-1)); nSizeLarge*(1:nFields)];
        nodeArrayLarge = cell2mat(cellfun(@(x) str2double(arrayfun(@(y, z) x(y:z), posLarge(1, :), posLarge(2, :), 'UniformOutput', false)), largeFormatCards, 'UniformOutput', false));
    else
        nodeArrayLarge = [];
    end
    nodeArray = [nodeArrayLarge; nodeArrayShort];

    nodes.id = nodeArray(:, 1);
    nodes.position = nodeArray(:, 3:5);
end

end