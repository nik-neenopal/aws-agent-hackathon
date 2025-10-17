import React, { useState } from "react";
export default function UploadAndResults() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState([]);
  const [errorModal, setErrorModal] = useState({ visible: false, message: "" });
  const LAMBDA_URL = process.env.REACT_APP_LAMBDA_UPLOAD_URL;
  const RESULTS_URL = process.env.REACT_APP_LAMBDA_RESULTS_URL;
  
  const REQUIRED_COLUMNS = [
    "series_id",
    "master_title",
    "release_year",
    "runtime",
    "program_type_name",
    "category_name",
    "episode_title",
    "episode_number",
    "credit_name",
    "rating_description",
    "rating_type_name",
  ];

  const handleFileChange = (e) => {
  const selectedFile = e.target.files[0];
  if (!selectedFile) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    const text = event.target.result;
    const lines = text.split("\n").map(line => line.trim()).filter(line => line !== "");
    
    //  Check number of rows (exclude header)
    const dataRowsCount = lines.length - 1;
if (dataRowsCount > 5) {
  setErrorModal({
    visible: true,
    message: `Max number of rows allowed is 5. You uploaded ${dataRowsCount} rows. Please upload again.`,
  });
  setFile(null);
  e.target.value = null;
  return;
}


    //  Check column headers
    const headers = lines[0].split(",").map(h => h.trim().replace(/\r/g, ""));
    const missing = REQUIRED_COLUMNS.filter(col => !headers.includes(col));

    if (missing.length > 0) {
      setErrorModal({
        visible: true,
        message: `Invalid CSV format. Missing columns: ${missing.join(", ")}\nPlease upload a file with all required columns.`,
      });
      setFile(null);
      e.target.value = null; 
      return;
    }


    //  File is valid
    setFile(selectedFile);
  };

  reader.readAsText(selectedFile);
};


  const uploadFile = async () => {
    if (!file) return alert("Please choose a CSV file first.");
    setUploading(true);
    try {
      // Step 1: POST to Lambda to get pre-signed S3 URL
      const lambdaResponse = await fetch(LAMBDA_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: file.name }),
      });
      if (!lambdaResponse.ok) throw new Error("Failed to get pre-signed URL");
      const { url: presignedUrl } = await lambdaResponse.json();
      
      // Step 2: PUT file to S3 using pre-signed URL
      const s3Response = await fetch(presignedUrl, {
        method: "PUT",
        body: file,
        headers: {
          "Content-Type": "text/csv",
        },
      });
      if (!s3Response.ok) throw new Error("Failed to upload file to S3");
      setUploading(false);
      setProcessing(true);
      
      // Step 3: Wait for processing (simulate 5 minutes)
      setTimeout(async () => {
        setProcessing(false);
        // Fetch results JSON
        const res = await fetch(RESULTS_URL);
        if (!res.ok) throw new Error("Failed to fetch results");
        const data = await res.json();
        console.log("Fetched results:", data); 
        setResults(data.rows || []);
      }, 5 * 60 * 1000); // 5 minutes
    } catch (err) {
      setUploading(false);
      setProcessing(false);
      alert("Upload failed: " + err.message);
    }
  };

    //  Function to trigger CSV sample download
  const downloadSampleCSV = () => {
  const sampleData = [
    [
      "series_id",
      "master_title",
      "release_year",
      "runtime",
      "program_type_name",
      "category_name",
      "episode_title",
      "episode_number",
      "credit_name",
      "rating_description",
      "rating_type_name",
    ],
    [
      "S001",
      "Stranger Things",
      "2016",
      "50",
      "Series",
      "Science Fiction",
      "Chapter One: The Vanishing of Will Byers",
      "1",
      "The Duffer Brothers",
      "TV-14",
      "V, L, S",
    ],
    [
      "S002",
      "Stranger Things",
      "2016",
      "50",
      "Series",
      "Science Fiction",
      "Chapter Two: The Weirdo on Maple Street",
      "2",
      "Millie Bobby Brown",
      "TV-14",
      "V, L, S",
    ],
    [
      "S003",
      "Stranger Things",
      "2016",
      "50",
      "Series",
      "Science Fiction",
      "Chapter Three: Holly, Jolly",
      "3",
      "",
      "TV-14",
      "V, L, S",
    ],
    [
      "S004",
      "The Crown",
      "2016",
      "59",
      "Series",
      "Historical Drama",
      "Wolferton Splash",
      "1",
      "Peter Morgan",
      "TV-MA",
      "L, S",
    ],
    [
      "S005",
      "The Crown",
      "2016",
      "",
      "Series",
      "",
      "Hyde Park Corner",
      "2",
      "Claire Foy",
      "TV-MA",
      "L, S",
    ],
    [
      "S006",
      "The Crown",
      "",
      "59",
      "Series",
      "Historical Drama",
      "Windsor",
      "3",
      "Matt Smith",
      "TV-MA",
      "",
    ],
    [
      "S007",
      "Black Mirror",
      "2011",
      "62",
      "Series",
      "Anthology",
      "The National Anthem",
      "1",
      "Charlie Brooker",
      "",
      "L, V",
    ],
    [
      "S008",
      "Black Mirror",
      "2011",
      "62",
      "Series",
      "Anthology",
      "",
      "2",
      "Annabel Jones",
      "TV-MA",
      "L, V",
    ],
    [
      "S009",
      "Black Mirror",
      "2011",
      "62",
      "Series",
      "Anthology",
      "The Entire History of You",
      "3",
      "Jesse Armstrong",
      "TV-MA",
      "L, V",
    ],
  ];

  const csvContent =
  "data:text/csv;charset=utf-8," +
  sampleData
    .map((row) =>
      row
        .map((val) => {
          if (val === null || val === undefined) return "";
          // Wrap in quotes if value contains comma or newline
          return /,|\n/.test(val) ? `"${val}"` : val;
        })
        .join(",")
    )
    .join("\n");

  const encodedUri = encodeURI(csvContent);

  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "sample_metadata.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};


  return (
    <div className="flex flex-col items-center px-6 py-12">
      <h1 className="text-4xl font-bold mb-4">Data Enrichment Agent</h1>
      <div className="flex gap-4 mb-12">
        <label className="cursor-pointer bg-red-600 text-white px-6 py-3 rounded-lg">
          {file ? file.name : "Choose CSV File"}
          <input
            type="file"
            accept=".csv"
            className="hidden"
            onChange={handleFileChange}          />
        </label>
        <button
          onClick={uploadFile}
          disabled={uploading || processing}
          className="bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg"
        >
          {uploading ? "Uploading..." : "Start Processing"}
        </button>
      </div>

      {/* Sample CSV Download Section */}
      <div className="mb-12 text-center">
        <p className="text-gray-600 mb-2">
          Download a sample CSV to see the format.
        </p>
        <button
          onClick={downloadSampleCSV}
          className="bg-gray-200 hover:bg-gray-300 text-gray-800 px-5 py-2 rounded-lg shadow-sm"
        >
          Download Sample CSV
        </button>
      </div>

        {/* Instructions Section */}
  <div className="mb-6 w-full max-w-4xl bg-gray-50 p-6 rounded-lg shadow-sm">
    <h3 className="text-lg font-semibold mb-4 text-center">
      Instructions
    </h3>
    <p className="text-gray-700 mb-4 text-center">
      The input file needs to follow specific structure as shown in the sample CSV below. The schema of the structure is as follows:
    </p>
    <div className="overflow-x-auto">
      <table className="min-w-full text-left border border-gray-300 rounded-lg">
        <thead className="bg-gray-200 text-gray-800">
          <tr>
            <th className="px-4 py-2 border-b">Column Name</th>
            <th className="px-4 py-2 border-b">Description</th>
          </tr>
        </thead>
        <tbody className="text-gray-700">
          <tr className="border-b hover:bg-gray-50">
            <td className="px-4 py-2">series_id</td>
            <td className="px-4 py-2">Unique identifier for each episode in the series (String)</td>
          </tr>
          <tr className="border-b hover:bg-gray-50">
            <td className="px-4 py-2">master_title</td>
            <td className="px-4 py-2">Name of the main TV series (String)</td>
          </tr>
          <tr className="border-b hover:bg-gray-50">
            <td className="px-4 py-2">release_year</td>
            <td className="px-4 py-2">Year the series or episode was released (Numeric)</td>
          </tr>
          <tr className="border-b hover:bg-gray-50">
            <td className="px-4 py-2">runtime</td>
            <td className="px-4 py-2">Duration of the episode in minutes (Numeric)</td>
          </tr>
          <tr className="border-b hover:bg-gray-50">
            <td className="px-4 py-2">program_type_name</td>
            <td className="px-4 py-2">Type of program, such as Series or Movie (String)</td>
          </tr>
          <tr className="border-b hover:bg-gray-50">
            <td className="px-4 py-2">category_name</td>
            <td className="px-4 py-2">Genre or category the show belongs to (String)</td>
          </tr>
          <tr className="border-b hover:bg-gray-50">
            <td className="px-4 py-2">episode_title</td>
            <td className="px-4 py-2">Title of the individual episode (String)</td>
          </tr>
          <tr className="border-b hover:bg-gray-50">
            <td className="px-4 py-2">episode_number</td>
            <td className="px-4 py-2">Sequential number of the episode within the series (Numeric)</td>
          </tr>
          <tr className="border-b hover:bg-gray-50">
            <td className="px-4 py-2">credit_name</td>
            <td className="px-4 py-2">Name of the main credited person (e.g., director, actor) (String)</td>
          </tr>
          <tr className="border-b hover:bg-gray-50">
            <td className="px-4 py-2">rating_description</td>
            <td className="px-4 py-2">Content rating details indicating violence, language, etc (String)</td>
          </tr>
          <tr className="border-b hover:bg-gray-50">
            <td className="px-4 py-2">rating_type_name</td>
            <td className="px-4 py-2">Official rating classification (e.g., TV-14, PG-13) (String)</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

      {errorModal.visible && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-lg w-96 text-center">
            <h2 className="text-xl font-semibold mb-4">Error</h2>
            <p className="mb-6">{errorModal.message}</p>
            <button
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg"
              onClick={() => setErrorModal({ visible: false, message: "" })}
            >
              Close
            </button>
          </div>
        </div>
      )}

      {processing && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
          <div className="bg-white p-6 rounded-lg shadow-lg text-center">
            <h2 className="text-xl font-semibold mb-2">Processing your file...</h2>
            <p>This may take ~5 minutes. Please wait!</p>
          </div>
        </div>
      )}

      {Array.isArray(results) && results.length > 0 && (
        <div className="w-full max-w-6xl bg-white rounded-2xl shadow-md p-6 mt-6">
          <h2 className="text-2xl font-semibold mb-4 text-center">
            Final Processed Data
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full text-left border border-gray-200 rounded-lg">
              <thead className="bg-gray-100 text-gray-700">
                <tr>
                  {Object.keys(results[0]).map((col) => (
                    <th key={col} className="px-4 py-2 border-b">
                      {col.toUpperCase()}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.map((row, idx) => (
                  <tr key={idx} className="hover:bg-gray-50">
                    {Object.values(row).map((val, i) => (
                      <td key={i} className="px-4 py-2 border-b">
                        {val !== null ? val.toString() : "-"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}