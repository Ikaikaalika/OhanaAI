import React, { useEffect, useRef, useState } from 'react';
import { parseGedcom } from './gedcom-parser';
import Tree from 'react-d3-tree';

const FanChart = ({ gedcomContent }) => {
  const treeContainerRef = useRef(null);
  const [translate, setTranslate] = React.useState({ x: 0, y: 0 });
  const [predictionMessage, setPredictionMessage] = useState('');

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

  const handlePredictMissingParents = async () => {
    setPredictionMessage('Predicting missing parents...');

    try {
      // This is a placeholder for generating candidate pairs.
      // In a real scenario, this would involve more sophisticated logic
      // based on the parsed GEDCOM data to identify potential missing parents.
      const candidatePairs = []; 
      // Example: if gedcomData.individuals has at least two individuals, create a dummy pair
      const individualIds = Object.keys(gedcomData.individuals);
      if (individualIds.length >= 2) {
        candidatePairs.push([0, 1]); // Dummy indices for demonstration
      }

      if (candidatePairs.length === 0) {
        setPredictionMessage('No candidate pairs generated for prediction.');
        return;
      }

      const token = localStorage.getItem('token');
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          gedcom_file: gedcomContent,
          candidate_pairs: candidatePairs,
        }),
      });

      const data = await response.json();
      if (response.ok) {
        setPredictionMessage(`Prediction successful! Found ${data.predictions.length} potential parents.`);
        // Here you would integrate the predictions back into the GEDCOM data
        // and re-render the fan chart.
        console.log('Predictions:', data.predictions);
      } else {
        setPredictionMessage(data.detail || 'Prediction failed.');
      }
    } catch (error) {
      setPredictionMessage('Error during prediction.');
      console.error('Prediction error:', error);
    }
  };

  const treeData = {
    name: gedcomData.individuals['@I1@']?.name || 'Root',
    children: [],
  };

  return (
    <div style={{ width: '100%', height: '500px' }}>
      <div ref={treeContainerRef} style={{ width: '100%', height: '100%' }}>
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
      <div style={{ marginTop: '20px', textAlign: 'center' }}>
        <button onClick={handlePredictMissingParents}>
          Predict Missing Parents
        </button>
        <p>{predictionMessage}</p>
      </div>
    </div>
  );
};

export default FanChart;