import React, { useEffect, useRef } from 'react';
import { parseGedcom } from './gedcom-parser'; // We will create this parser
import Tree from 'react-d3-tree';

const FanChart = ({ gedcomContent }) => {
  const treeContainerRef = useRef(null);
  const [translate, setTranslate] = React.useState({ x: 0, y: 0 });

  useEffect(() => {
    if (treeContainerRef.current) {
      const dimensions = treeContainerRef.current.getBoundingClientRect();
      setTranslate({ x: dimensions.width / 2, y: dimensions.height / 2 });
    }
  }, []);

  if (!gedcomContent) {
    return <p>Upload a GEDCOM file to see the fan chart.</p>;
  }

  let gedcomData = {};
  try {
    gedcomData = parseGedcom(gedcomContent);
  } catch (error) {
    console.error("Error parsing GEDCOM content:", error);
    return <p>Error parsing GEDCOM content. Please check the file format.</p>;
  }

  // Transform gedcomData into a format suitable for react-d3-tree
  // This is a simplified example; a real implementation would be more complex
  const treeData = {
    name: gedcomData.individuals['@I1@']?.name || 'Root',
    children: [],
  };

  return (
    <div ref={treeContainerRef} style={{ width: '100%', height: '500px' }}>
      <Tree
        data={treeData}
        translate={translate}
        orientation="radial"
        nodeSize={{ x: 120, y: 120 }}
        separation={{ siblings: 1, nonSiblings: 2 }}
        pathFunc="step"
        zoomable={true}
        draggable={true}
        collapsible={true}
        renderCustomNodeElement={({ nodeDatum, toggleNode }) => (
          <g>
            <circle r={15} onClick={toggleNode} fill="lightblue" />
            <text fill="black" strokeWidth="1" x="20" style={{ fontSize: '10px' }}>
              {nodeDatum.name}
            </text>
          </g>
        )}
      />
    </div>
  );
};

export default FanChart;
