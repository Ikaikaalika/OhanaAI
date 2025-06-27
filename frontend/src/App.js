import React, { useState } from 'react';
import './App.css';
import FanChart from './FanChart';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState('');
  const [gedcomFile, setGedcomFile] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [selectedGedcomContent, setSelectedGedcomContent] = useState('');

  const handleLogin = async (e) => {
    e.preventDefault();
    setMessage('');
    try {
      const response = await fetch('/api/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          username: username,
          password: password,
        }).toString(),
      });
      const data = await response.json();
      if (response.ok) {
        localStorage.setItem('token', data.access_token);
        setIsLoggedIn(true);
        setMessage('Login successful!');
        fetchUploadedFiles();
      } else {
        setMessage(data.detail || 'Login failed.');
      }
    } catch (error) {
      setMessage('Error during login.');
      console.error('Login error:', error);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setMessage('');
    try {
      const response = await fetch('/api/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      const data = await response.json();
      if (response.ok) {
        setMessage('Registration successful! Please log in.');
      } else {
        setMessage(data.detail || 'Registration failed.');
      }
    } catch (error) {
      setMessage('Error during registration.');
      console.error('Registration error:', error);
    }
  };

  const handleFileUpload = async (e) => {
    e.preventDefault();
    if (!gedcomFile) {
      setMessage('Please select a GEDCOM file.');
      return;
    }

    setMessage('');
    const formData = new FormData();
    formData.append('file', gedcomFile);

    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/gedcom/upload', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setMessage('File uploaded successfully!');
        setGedcomFile(null);
        fetchUploadedFiles();
      } else {
        setMessage(data.detail || 'File upload failed.');
      }
    } catch (error) {
      setMessage('Error during file upload.');
      console.error('Upload error:', error);
    }
  };

  const fetchUploadedFiles = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/gedcom/list', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      const data = await response.json();
      if (response.ok) {
        setUploadedFiles(data);
      } else {
        setMessage(data.detail || 'Failed to fetch uploaded files.');
      }
    } catch (error) {
      setMessage('Error fetching uploaded files.');
      console.error('Fetch files error:', error);
    }
  };

  const handleViewGedcom = async (fileId) => {
    setMessage('');
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`/api/gedcom/${fileId}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      if (response.ok) {
        const gedcomText = await response.text();
        setSelectedGedcomContent(gedcomText);
      } else {
        setMessage(data.detail || 'Failed to fetch GEDCOM content.');
      }
    } catch (error) {
      setMessage('Error fetching GEDCOM content.');
      console.error('Fetch GEDCOM error:', error);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setIsLoggedIn(false);
    setUsername('');
    setPassword('');
    setMessage('Logged out.');
    setUploadedFiles([]);
    setSelectedGedcomContent('');
  };

  const handleDeleteAllData = async () => {
    if (!window.confirm('Are you sure you want to delete all your GEDCOM data? This action cannot be undone.')) {
      return;
    }
    setMessage('');
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/gedcom/delete_all', {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      const data = await response.json();
      if (response.ok) {
        setMessage(data.message);
        setUploadedFiles([]);
        setSelectedGedcomContent('');
      } else {
        setMessage(data.detail || 'Failed to delete data.');
      }
    } catch (error) {
      setMessage('Error deleting data.');
      console.error('Delete data error:', error);
    }
  };

  const handleDeleteAccount = async () => {
    if (!window.confirm('Are you sure you want to delete your account? This action cannot be undone and will delete all your data.')) {
      return;
    }
    setMessage('');
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/users/delete_me', {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      const data = await response.json();
      if (response.ok) {
        setMessage(data.message);
        handleLogout(); // Log out after account deletion
      } else {
        setMessage(data.detail || 'Failed to delete account.');
      }
    } catch (error) {
      setMessage('Error deleting account.');
      console.error('Delete account error:', error);
    }
  };

  const handleDownloadFilledGedcom = async (fileId, filename) => {
    setMessage('');
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`/api/gedcom/${fileId}/filled`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      if (response.ok) {
        const gedcomText = await response.text();
        const blob = new Blob([gedcomText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `filled_${filename}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        setMessage('Filled GEDCOM file downloaded.');
      } else {
        setMessage(data.detail || 'Failed to download filled GEDCOM.');
      }
    } catch (error) {
      setMessage('Error downloading filled GEDCOM.');
      console.error('Download filled GEDCOM error:', error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>OhanaAI</h1>
        {message && <p>{message}</p>}

        {!isLoggedIn ? (
          <div>
            <h2>Login / Register</h2>
            <form onSubmit={handleLogin}>
              <input
                type="text"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <button type="submit">Login</button>
              <button type="button" onClick={handleRegister}>Register</button>
            </form>
          </div>
        ) : (
          <div>
            <h2>Welcome, {username}!</h2>
            <button onClick={handleLogout}>Logout</button>
            <button onClick={handleDeleteAllData}>Delete All My Data</button>
            <button onClick={handleDeleteAccount}>Delete My Account</button>

            <h3>Upload GEDCOM File</h3>
            <form onSubmit={handleFileUpload}>
              <input
                type="file"
                accept=".ged,.gedcom"
                onChange={(e) => setGedcomFile(e.target.files[0])}
              />
              <button type="submit">Upload</button>
            </form>

            <h3>Your GEDCOM Files</h3>
            {uploadedFiles.length === 0 ? (
              <p>No files uploaded yet.</p>
            ) : (
              <ul>
                {uploadedFiles.map((file) => (
                  <li key={file.id}>
                    {file.filename} (Status: {file.status})
                    <button onClick={() => handleViewGedcom(file.id)}>View GEDCOM</button>
                    <button onClick={() => handleDownloadFilledGedcom(file.id, file.filename)}>Download Filled GEDCOM</button>
                  </li>
                ))}
              </ul>
            )}

            {selectedGedcomContent && (
              <div className="gedcom-viewer">
                <h3>GEDCOM Content</h3>
                <pre>{selectedGedcomContent}</pre>
              </div>
            )}

            <div className="fan-chart-container">
              <FanChart gedcomContent={selectedGedcomContent} />
            </div>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
