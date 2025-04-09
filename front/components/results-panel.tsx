import { Card } from "@/components/ui/card"
import { AlertTriangle, FileText, User, Shield, Clock } from "lucide-react"

interface ResultsPanelProps {
	imageData: string | null
	isProcessing: boolean
	person: Suspect | undefined
}

export default function ResultsPanel({ imageData, person, isProcessing }: ResultsPanelProps) {
	// Get threat level color
	const getThreatColor = () => {
		switch (person?.threatLevel) {
			case "LOW":
				return "text-green-500"
			case "MODERATE":
				return "text-yellow-500"
			case "HIGH":
				return "text-orange-500"
			case "CRITICAL":
				return "text-red-500"
			default:
				return "text-gray-500"
		}
	}

	// Get threat level border color
	const getThreatBorderColor = () => {
		switch (person?.threatLevel) {
			case "LOW":
				return "border-green-500"
			case "MODERATE":
				return "border-yellow-500"
			case "HIGH":
				return "border-orange-500"
			case "CRITICAL":
				return "border-red-500"
			default:
				return "border-gray-500"
		}
	}

	return (
		<Card className="lg:col-span-2 bg-gray-900 border-blue-900">
			<div className="p-4 border-b border-blue-900 flex justify-between items-center">
				<div className="flex items-center">
					<div className="h-2 w-2 rounded-full bg-blue-500 mr-2"></div>
					<h3 className="text-sm font-mono uppercase tracking-wider text-blue-500">Subject Analysis</h3>
				</div>
			</div>

			<div className="p-4 space-y-4">
				{imageData ? (
					<div className="border border-blue-900 rounded overflow-hidden">
						<img
							src={imageData || "/placeholder.svg"}
							alt="Subject"
							className="w-full h-auto object-contain max-h-[200px]"
						/>
					</div>
				) : (
					<div className="border border-blue-900 rounded bg-gray-950 h-[200px] flex items-center justify-center">
						<p className="text-blue-500 font-mono text-sm">NO SUBJECT CAPTURED</p>
					</div>
				)}

				{isProcessing ? (
					<div className="border border-blue-900 rounded p-4 bg-gray-950">
						<div className="flex flex-col items-center justify-center space-y-2">
							<div className="w-16 h-16 border-4 border-t-blue-500 border-blue-900/30 rounded-full animate-spin"></div>
							<p className="text-blue-500 font-mono text-sm">ANALYZING SUBJECT BIOMETRICS</p>
							<div className="w-full bg-gray-800 h-2 mt-2">
								<div className="bg-blue-500 h-2 animate-pulse" style={{ width: "60%" }}></div>
							</div>
						</div>
					</div>
				) : person ? (
					<div className={`border rounded p-4 bg-gray-950 ${getThreatBorderColor()}`}>
						<div className="flex justify-between items-start mb-4">
							<div className="flex items-center">
								<User className="h-5 w-5 text-blue-500 mr-2" />
								<h4 className="font-mono text-white">SUBJECT IDENTIFIED</h4>
							</div>
							<div
								className={`px-2 py-1 rounded font-mono text-xs ${getThreatColor()} border ${getThreatBorderColor()}`}
							>
								THREAT LEVEL: {person.threatLevel}
							</div>
						</div>

						<div className="space-y-3 font-mono">
							<div className="grid grid-cols-2 gap-2 text-xs">
								<div className="text-blue-400">NAME:</div>
								<div className="text-white uppercase">{person.name}</div>

								<div className="text-blue-400">DATABASE MATCH:</div>
								<div className="text-white">{Math.round(person.confidence*100)/100}%</div>

								<div className="text-blue-400">LAST SEEN:</div>
								<div className="text-white">{new Date().toLocaleDateString()}</div>

								<div className="text-blue-400">CRIMES:</div>
								<div className="text-white">{person.crimes.join(", ")}</div>
							</div>

							<div className="pt-2 border-t border-blue-900 mt-2 flex items-center">
								<AlertTriangle className="h-4 w-4 text-yellow-500 mr-1" />
								<span className="text-yellow-500 text-xs">FURTHER INVESTIGATION REQUIRED</span>
							</div>
						</div>
					</div>
				) : imageData ? (
					<div className="border border-blue-900 rounded p-4 bg-gray-950">
						<div className="flex items-center justify-center h-24">
							<p className="text-blue-500 font-mono text-sm">AWAITING ANALYSIS</p>
						</div>
					</div>
				) : (
					<div className="border border-blue-900 rounded p-4 bg-gray-950">
						<div className="flex items-center justify-center h-24">
							<p className="text-blue-500 font-mono text-sm">NO DATA AVAILABLE</p>
						</div>
					</div>
				)}

				<div className="border border-blue-900 rounded p-3 bg-gray-950">
					<div className="flex items-center mb-2">
						<FileText className="h-4 w-4 text-blue-500 mr-2" />
						<h4 className="font-mono text-sm text-blue-500">SYSTEM LOG</h4>
					</div>
					<div className="space-y-1 text-xs font-mono text-gray-400">
						<p>[{new Date().toLocaleTimeString()}] System initialized</p>
						{imageData && <p>[{new Date().toLocaleTimeString()}] Subject image captured</p>}
						{person && (
							<p>
								[{new Date().toLocaleTimeString()}] Subject identified: {person.name}
							</p>
						)}
						{person && (
							<p>
								[{new Date().toLocaleTimeString()}] Threat assessment: {person.threatLevel}
							</p>
						)}
					</div>
				</div>
			</div>

			<div className="p-4 border-t border-blue-900 flex justify-between items-center">
				<div className="flex items-center text-xs font-mono text-blue-400">
					<Shield className="h-3 w-3 mr-1" />
					<span>SECURE TERMINAL</span>
				</div>
				<div className="flex items-center text-xs font-mono text-blue-400">
					<Clock className="h-3 w-3 mr-1" />
					<span>{new Date().toLocaleTimeString()}</span>
				</div>
			</div>
		</Card>
	)
}
