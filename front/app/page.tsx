"use client"

import { useState } from "react"
import axios from "axios"
import { Card } from "@/components/ui/card"
import CameraCapture from "@/components/camera-capture"
import { Shield, AlertTriangle } from "lucide-react"
import Header from "@/components/header"
import ResultsPanel from "@/components/results-panel"

// Get API URL from environment variable or use default
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
	const [imageData, setImageData] = useState<string | null>(null)
	const [isProcessing, setIsProcessing] = useState(false)
	const [person, setPerson] = useState<Suspect | undefined>(undefined)

	const handleCapture = (imageSrc: string) => {
		setImageData(imageSrc)
		processImage(imageSrc)
	}

	const processImage = async (imageSrc: string) => {
		setIsProcessing(true)
		try {
			// Convert base64 to blob
			const base64Response = await fetch(imageSrc);
			const blob = await base64Response.blob();

			// Create FormData and append the image
			const formData = new FormData();
			formData.append('image', blob, 'capture.jpg');

			const response = await axios.post(`${API_URL}/upload-image/`, formData, {
				headers: {
					'Content-Type': 'multipart/form-data',
				},
			});

			const data = response.data as Suspect[];

			if(data.length > 0)
				setPerson(data[0])
			else
				setPerson(undefined)
		} catch (error) {
			console.error("Error processing image:", error)
		} finally {
			setIsProcessing(false)
		}
	}

	return (
		<main className="min-h-screen bg-black text-white">
			<Header />

			<div className="container mx-auto py-6 px-4">
				<div className="flex items-center justify-between mb-6 border-b border-blue-900 pb-2">
					<div className="flex items-center">
						<Shield className="h-5 w-5 text-blue-500 mr-2" />
						<h2 className="text-lg font-mono uppercase tracking-wider">Facial Recognition System v3.4.2</h2>
					</div>
					<div className="flex items-center">
						<div className="h-2 w-2 rounded-full bg-green-500 mr-2 animate-pulse"></div>
						<span className="text-xs font-mono text-green-500">SYSTEM ACTIVE</span>
					</div>
				</div>

				<div className="grid lg:grid-cols-5 gap-6">
					<Card className="lg:col-span-3 bg-gray-900 border-blue-900">
						<div className="p-4 border-b border-blue-900 flex justify-between items-center">
							<div className="flex items-center">
								<div className="h-2 w-2 rounded-full bg-blue-500 mr-2"></div>
								<h3 className="text-sm font-mono uppercase tracking-wider text-blue-500">Surveillance Feed</h3>
							</div>
							<div className="text-xs font-mono text-blue-400">{new Date().toLocaleTimeString()} | SECURE CHANNEL</div>
						</div>
						<div className="p-4" style={{ minHeight: "350px" }}>
							<CameraCapture onCapture={handleCapture} />
						</div>
					</Card>

					<ResultsPanel
						imageData={imageData}
						person={person}
						isProcessing={isProcessing}
					/>
				</div>

				<div className="mt-6 text-xs font-mono text-gray-500 border-t border-gray-800 pt-4">
					<div className="flex items-center">
						<AlertTriangle className="h-3 w-3 mr-1 text-yellow-600" />
						<span>CONFIDENTIAL: AUTHORIZED PERSONNEL ONLY - LEVEL 4 CLEARANCE REQUIRED</span>
					</div>
				</div>
			</div>
		</main>
	)
}
