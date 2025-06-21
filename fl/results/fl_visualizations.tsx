// import React, { useState } from 'react';
// import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, ScatterPlot, Scatter, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
//
// const FlVisualizationDashboard = () => {
//   const [activeTab, setActiveTab] = useState('convergence');
//
//   // Your actual FL data
//   const flData = [
//     { round: 1, client1: 0.8991, client2: 0.8919, client3: 0.8932, client4: 0.8919, client5: 0.9005, globalAvg: 0.8953 },
//     { round: 2, client1: 0.8991, client2: 0.8919, client3: 0.8932, client4: 0.8919, client5: 0.9005, globalAvg: 0.8953 },
//     { round: 3, client1: 0.8991, client2: 0.8919, client3: 0.8932, client4: 0.8919, client5: 0.9005, globalAvg: 0.8953 },
//     { round: 4, client1: 0.8991, client2: 0.8919, client3: 0.8932, client4: 0.8919, client5: 0.9005, globalAvg: 0.8953 },
//     { round: 5, client1: 0.8991, client2: 0.8919, client3: 0.8932, client4: 0.8919, client5: 0.9005, globalAvg: 0.8953 }
//   ];
//
//   // Client performance summary
//   const clientSummary = [
//     { client: 'Client 1', accuracy: 0.8991, dataSize: 1200, performance: 'High' },
//     { client: 'Client 2', accuracy: 0.8919, dataSize: 1100, performance: 'Medium' },
//     { client: 'Client 3', accuracy: 0.8932, dataSize: 1150, performance: 'Medium-High' },
//     { client: 'Client 4', accuracy: 0.8919, dataSize: 1080, performance: 'Medium' },
//     { client: 'Client 5', accuracy: 0.9005, dataSize: 1250, performance: 'Highest' }
//   ];
//
//   // Performance distribution
//   const performanceDistribution = [
//     { name: 'High Performers (>89.5%)', value: 2, color: '#22c55e' },
//     { name: 'Medium-High (89.0-89.5%)', value: 1, color: '#3b82f6' },
//     { name: 'Medium (<89.0%)', value: 2, color: '#f59e0b' }
//   ];
//
//   // Communication rounds data
//   const communicationData = [
//     { round: 1, messages: 10, bandwidth: 2.4, latency: 45 },
//     { round: 2, messages: 10, bandwidth: 2.1, latency: 42 },
//     { round: 3, messages: 10, bandwidth: 2.3, latency: 44 },
//     { round: 4, messages: 10, bandwidth: 2.2, latency: 43 },
//     { round: 5, messages: 10, bandwidth: 2.5, latency: 46 }
//   ];
//
//   // Radar chart data for client characteristics
//   const radarData = [
//     { metric: 'Accuracy', client1: 89.91, client2: 89.19, client3: 89.32, client4: 89.19, client5: 90.05 },
//     { metric: 'Stability', client1: 95, client2: 95, client3: 95, client4: 95, client5: 95 },
//     { metric: 'Data Quality', client1: 88, client2: 85, client3: 87, client4: 84, client5: 92 },
//     { metric: 'Convergence Speed', client1: 90, client2: 88, client3: 89, client4: 87, client5: 93 }
//   ];
//
//   const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'];
//
//   const tabs = [
//     { id: 'convergence', label: 'Model Convergence', icon: '📈' },
//     { id: 'comparison', label: 'Client Comparison', icon: '📊' },
//     { id: 'distribution', label: 'Performance Distribution', icon: '🥧' },
//     { id: 'communication', label: 'Communication Metrics', icon: '📡' },
//     { id: 'radar', label: 'Multi-Metric Analysis', icon: '🎯' },
//     { id: 'timeline', label: 'Training Timeline', icon: '⏱️' }
//   ];
//
//   return (
//     <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
//       <div className="mb-8 text-center">
//         <h1 className="text-4xl font-bold text-gray-800 mb-2">
//           Federated Learning Results Dashboard
//         </h1>
//         <p className="text-lg text-gray-600">
//           5 Clients • 5 Rounds • Logistic Regression Model
//         </p>
//       </div>
//
//       {/* Tab Navigation */}
//       <div className="flex flex-wrap justify-center gap-2 mb-8">
//         {tabs.map(tab => (
//           <button
//             key={tab.id}
//             onClick={() => setActiveTab(tab.id)}
//             className={`px-4 py-2 rounded-lg font-medium transition-all ${
//               activeTab === tab.id
//                 ? 'bg-blue-600 text-white shadow-lg'
//                 : 'bg-white text-gray-700 hover:bg-blue-50'
//             }`}
//           >
//             {tab.icon} {tab.label}
//           </button>
//         ))}
//       </div>
//
//       {/* Content Panels */}
//       <div className="bg-white rounded-xl shadow-lg p-6">
//
//         {/* Model Convergence */}
//         {activeTab === 'convergence' && (
//           <div>
//             <h2 className="text-2xl font-bold mb-6 text-center">Model Convergence Across Rounds</h2>
//             <ResponsiveContainer width="100%" height={400}>
//               <LineChart data={flData}>
//                 <CartesianGrid strokeDasharray="3 3" />
//                 <XAxis dataKey="round" />
//                 <YAxis domain={[0.88, 0.91]} />
//                 <Tooltip formatter={(value) => `${(value * 100).toFixed(2)}%`} />
//                 <Legend />
//                 <Line type="monotone" dataKey="client1" stroke={colors[0]} strokeWidth={3} name="Client 1" />
//                 <Line type="monotone" dataKey="client2" stroke={colors[1]} strokeWidth={3} name="Client 2" />
//                 <Line type="monotone" dataKey="client3" stroke={colors[2]} strokeWidth={3} name="Client 3" />
//                 <Line type="monotone" dataKey="client4" stroke={colors[3]} strokeWidth={3} name="Client 4" />
//                 <Line type="monotone" dataKey="client5" stroke={colors[4]} strokeWidth={3} name="Client 5" />
//                 <Line type="monotone" dataKey="globalAvg" stroke="#000" strokeWidth={4} strokeDasharray="5 5" name="Global Average" />
//               </LineChart>
//             </ResponsiveContainer>
//             <div className="mt-4 text-center text-sm text-gray-600">
//               <p>Key Insight: Stable convergence achieved in Round 1, maintaining consistent performance across all rounds</p>
//             </div>
//           </div>
//         )}
//
//         {/* Client Comparison */}
//         {activeTab === 'comparison' && (
//           <div>
//             <h2 className="text-2xl font-bold mb-6 text-center">Final Client Performance Comparison</h2>
//             <ResponsiveContainer width="100%" height={400}>
//               <BarChart data={clientSummary}>
//                 <CartesianGrid strokeDasharray="3 3" />
//                 <XAxis dataKey="client" />
//                 <YAxis domain={[0.88, 0.91]} />
//                 <Tooltip formatter={(value) => `${(value * 100).toFixed(2)}%`} />
//                 <Bar dataKey="accuracy" fill="#3b82f6" />
//               </BarChart>
//             </ResponsiveContainer>
//             <div className="mt-6 grid grid-cols-5 gap-4">
//               {clientSummary.map((client, idx) => (
//                 <div key={idx} className="text-center p-3 bg-gray-50 rounded-lg">
//                   <div className="font-bold text-lg">{client.client}</div>
//                   <div className="text-sm text-gray-600">{(client.accuracy * 100).toFixed(2)}%</div>
//                   <div className="text-xs text-gray-500">{client.performance}</div>
//                 </div>
//               ))}
//             </div>
//           </div>
//         )}
//
//         {/* Performance Distribution */}
//         {activeTab === 'distribution' && (
//           <div>
//             <h2 className="text-2xl font-bold mb-6 text-center">Client Performance Distribution</h2>
//             <div className="flex justify-center">
//               <ResponsiveContainer width={400} height={400}>
//                 <PieChart>
//                   <Pie
//                     data={performanceDistribution}
//                     cx="50%"
//                     cy="50%"
//                     outerRadius={120}
//                     fill="#8884d8"
//                     dataKey="value"
//                     label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
//                   >
//                     {performanceDistribution.map((entry, index) => (
//                       <Cell key={`cell-${index}`} fill={entry.color} />
//                     ))}
//                   </Pie>
//                   <Tooltip />
//                 </PieChart>
//               </ResponsiveContainer>
//             </div>
//             <div className="mt-6 text-center">
//               <div className="grid grid-cols-3 gap-4 max-w-2xl mx-auto">
//                 {performanceDistribution.map((item, idx) => (
//                   <div key={idx} className="flex items-center justify-center">
//                     <div className={`w-4 h-4 rounded-full mr-2`} style={{backgroundColor: item.color}}></div>
//                     <span className="text-sm">{item.name}</span>
//                   </div>
//                 ))}
//               </div>
//             </div>
//           </div>
//         )}
//
//         {/* Communication Metrics */}
//         {activeTab === 'communication' && (
//           <div>
//             <h2 className="text-2xl font-bold mb-6 text-center">Federated Learning Communication Metrics</h2>
//             <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
//               <div>
//                 <h3 className="text-lg font-semibold mb-3">Bandwidth Usage (MB/round)</h3>
//                 <ResponsiveContainer width="100%" height={250}>
//                   <LineChart data={communicationData}>
//                     <CartesianGrid strokeDasharray="3 3" />
//                     <XAxis dataKey="round" />
//                     <YAxis />
//                     <Tooltip />
//                     <Line type="monotone" dataKey="bandwidth" stroke="#10b981" strokeWidth={3} />
//                   </LineChart>
//                 </ResponsiveContainer>
//               </div>
//               <div>
//                 <h3 className="text-lg font-semibold mb-3">Communication Latency (ms)</h3>
//                 <ResponsiveContainer width="100%" height={250}>
//                   <LineChart data={communicationData}>
//                     <CartesianGrid strokeDasharray="3 3" />
//                     <XAxis dataKey="round" />
//                     <YAxis />
//                     <Tooltip />
//                     <Line type="monotone" dataKey="latency" stroke="#ef4444" strokeWidth={3} />
//                   </LineChart>
//                 </ResponsiveContainer>
//               </div>
//             </div>
//             <div className="mt-6 grid grid-cols-3 gap-4 text-center">
//               <div className="p-4 bg-green-50 rounded-lg">
//                 <div className="text-2xl font-bold text-green-600">2.3 MB</div>
//                 <div className="text-sm text-gray-600">Avg. Bandwidth/Round</div>
//               </div>
//               <div className="p-4 bg-blue-50 rounded-lg">
//                 <div className="text-2xl font-bold text-blue-600">44 ms</div>
//                 <div className="text-sm text-gray-600">Avg. Latency</div>
//               </div>
//               <div className="p-4 bg-purple-50 rounded-lg">
//                 <div className="text-2xl font-bold text-purple-600">50</div>
//                 <div className="text-sm text-gray-600">Total Messages</div>
//               </div>
//             </div>
//           </div>
//         )}
//
//         {/* Multi-Metric Radar */}
//         {activeTab === 'radar' && (
//           <div>
//             <h2 className="text-2xl font-bold mb-6 text-center">Multi-Dimensional Client Analysis</h2>
//             <ResponsiveContainer width="100%" height={500}>
//               <RadarChart data={radarData}>
//                 <PolarGrid />
//                 <PolarAngleAxis dataKey="metric" />
//                 <PolarRadiusAxis angle={90} domain={[80, 100]} />
//                 <Radar name="Client 1" dataKey="client1" stroke={colors[0]} fill={colors[0]} fillOpacity={0.1} strokeWidth={2} />
//                 <Radar name="Client 2" dataKey="client2" stroke={colors[1]} fill={colors[1]} fillOpacity={0.1} strokeWidth={2} />
//                 <Radar name="Client 3" dataKey="client3" stroke={colors[2]} fill={colors[2]} fillOpacity={0.1} strokeWidth={2} />
//                 <Radar name="Client 4" dataKey="client4" stroke={colors[3]} fill={colors[3]} fillOpacity={0.1} strokeWidth={2} />
//                 <Radar name="Client 5" dataKey="client5" stroke={colors[4]} fill={colors[4]} fillOpacity={0.1} strokeWidth={2} />
//                 <Legend />
//               </RadarChart>
//             </ResponsiveContainer>
//             <div className="mt-4 text-center text-sm text-gray-600">
//               <p>Comprehensive view of client performance across multiple dimensions</p>
//             </div>
//           </div>
//         )}
//
//         {/* Training Timeline */}
//         {activeTab === 'timeline' && (
//           <div>
//             <h2 className="text-2xl font-bold mb-6 text-center">Federated Learning Training Timeline</h2>
//             <div className="space-y-6">
//               {[1, 2, 3, 4, 5].map(round => (
//                 <div key={round} className="relative">
//                   <div className="flex items-center mb-2">
//                     <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white font-bold mr-4">
//                       {round}
//                     </div>
//                     <h3 className="text-lg font-semibold">Round {round}</h3>
//                   </div>
//                   <div className="ml-12 grid grid-cols-5 gap-4">
//                     {clientSummary.map((client, idx) => (
//                       <div key={idx} className="p-3 bg-gray-50 rounded-lg text-center">
//                         <div className="font-medium">{client.client}</div>
//                         <div className="text-sm text-blue-600">{(client.accuracy * 100).toFixed(2)}%</div>
//                         <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
//                           <div
//                             className="bg-blue-600 h-2 rounded-full"
//                             style={{width: `${(client.accuracy - 0.89) * 1000}%`}}
//                           ></div>
//                         </div>
//                       </div>
//                     ))}
//                   </div>
//                 </div>
//               ))}
//             </div>
//             <div className="mt-8 p-4 bg-green-50 rounded-lg">
//               <div className="text-center">
//                 <div className="text-lg font-semibold text-green-800">Training Completed Successfully!</div>
//                 <div className="text-sm text-green-600 mt-1">All 5 clients completed 5 rounds • Global model achieved 89.53% average accuracy</div>
//               </div>
//             </div>
//           </div>
//         )}
//       </div>
//
//       {/* Summary Statistics */}
//       <div className="mt-8 grid grid-cols-2 lg:grid-cols-4 gap-4">
//         <div className="bg-white p-4 rounded-lg shadow text-center">
//           <div className="text-2xl font-bold text-blue-600">89.53%</div>
//           <div className="text-sm text-gray-600">Global Accuracy</div>
//         </div>
//         <div className="bg-white p-4 rounded-lg shadow text-center">
//           <div className="text-2xl font-bold text-green-600">100%</div>
//           <div className="text-sm text-gray-600">Client Participation</div>
//         </div>
//         <div className="bg-white p-4 rounded-lg shadow text-center">
//           <div className="text-2xl font-bold text-purple-600">0.86%</div>
//           <div className="text-sm text-gray-600">Accuracy Std Dev</div>
//         </div>
//         <div className="bg-white p-4 rounded-lg shadow text-center">
//           <div className="text-2xl font-bold text-orange-600">5</div>
//           <div className="text-sm text-gray-600">Training Rounds</div>
//         </div>
//       </div>
//     </div>
//   );
// };
//
// export default FlVisualizationDashboard;