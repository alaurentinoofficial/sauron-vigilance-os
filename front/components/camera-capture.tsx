"use client"

import { useRef, useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Camera, RefreshCw, AlertCircle } from "lucide-react"

interface CameraCaptureProps {
	onCapture: (imageSrc: string) => void
}

export default function CameraCapture({ onCapture }: CameraCaptureProps) {
	const videoRef = useRef<HTMLVideoElement>(null)
	const canvasRef = useRef<HTMLCanvasElement>(null)
	const [stream, setStream] = useState<MediaStream | null>(null)
	const [error, setError] = useState<string | null>(null)
	const [isCameraActive, setIsCameraActive] = useState(false)

	const startCamera = async () => {
		try {
			setError(null)
			const mediaStream = await navigator.mediaDevices.getUserMedia({
				video: true,
			})

			setStream(mediaStream)
			setIsCameraActive(true)

			if (videoRef.current) {
			}
		} catch (err) {
			console.error("Error accessing camera:", err)
			setError("ACCESS DENIED: CAMERA PERMISSION REQUIRED")
		}
	}

	const stopCamera = () => {
		if (stream) {
			stream.getTracks().forEach((track) => track.stop())
			setStream(null)
			setIsCameraActive(false)
			if (videoRef.current) {
				videoRef.current.srcObject = null
			}
		}
	}

	const captureImage = () => {
		if (videoRef.current && canvasRef.current) {
			console.log("Camera activated, stream attached to video element")
			const video = videoRef.current
			const canvas = canvasRef.current
			const context = canvas.getContext("2d")

			if (context) {
				// Set canvas dimensions to match video
				canvas.width = video.videoWidth
				canvas.height = video.videoHeight

				// Draw the current video frame to the canvas
				context.drawImage(video, 0, 0, canvas.width, canvas.height)

				// Convert canvas to data URL
				const imageSrc = canvas.toDataURL("image/jpeg")
				onCapture(imageSrc)
			}
		}
	}

	useEffect(() => {
		// Clean up on unmount
		return () => {
			stopCamera()
		}
	}, [])

	useEffect(() => {
		if (videoRef.current && isCameraActive) {
			videoRef.current.srcObject = stream
			videoRef.current.play().catch((e) => console.error("Error playing video:", e))

			videoRef.current.onloadedmetadata = () => {
				console.log("Video metadata loaded, stream should be visible")
			}

			videoRef.current.onplay = () => {
				console.log("Video is now playing")

				const fallbackElements = document.getElementsByClassName("camera-fallback")
				if (fallbackElements.length > 0) {
					; (fallbackElements[0] as HTMLElement).style.display = "none"
				}
			}
		}
	}, [stream, isCameraActive])

	console.log(isCameraActive)

	return (
		<div className="space-y-4">
			<div className="relative bg-black rounded-lg overflow-hidden border border-blue-900">
				{isCameraActive  ? (
					<>
						<video
							ref={videoRef}
							autoPlay
							playsInline
							className="w-full scale-x-[-1] h-[300px] object-cover"
							style={{ display: "block" }}
						/>

						<div className="absolute inset-0 flex items-center justify-center bg-black/80 z-10 camera-fallback">
							<p className="text-blue-500 font-mono text-sm text-center">
								If camera feed is not visible, please check your browser permissions.
								<br />
								<button
									className="mt-2 px-3 py-1 bg-blue-900 text-white rounded text-xs"
									onClick={() => {
										if (videoRef.current && videoRef.current.srcObject) {
											videoRef.current.play()
										}
									}}
								>
									RETRY CAMERA
								</button>
							</p>
						</div>

						{/* FBI-style overlays */}
						<div className="absolute inset-0 pointer-events-none">
							{/* Grid overlay */}
							<div className="w-full h-full grid grid-cols-12 grid-rows-12">
								{Array.from({ length: 12 }).map((_, i) => (
									<div key={`col-${i}`} className="border-r border-blue-500/20"></div>
								))}
								{Array.from({ length: 12 }).map((_, i) => (
									<div key={`row-${i}`} className="border-b border-blue-500/20"></div>
								))}
							</div>

							{/* Corner brackets */}
							<div className="absolute top-0 left-0 w-12 h-12 border-t-2 border-l-2 border-red-500"></div>
							<div className="absolute top-0 right-0 w-12 h-12 border-t-2 border-r-2 border-red-500"></div>
							<div className="absolute bottom-0 left-0 w-12 h-12 border-b-2 border-l-2 border-red-500"></div>
							<div className="absolute bottom-0 right-0 w-12 h-12 border-b-2 border-r-2 border-red-500"></div>

							{/* Center target */}
							<div className="absolute inset-0 flex items-center justify-center">
								<div className="border-2 border-red-500 w-[180px] h-[180px] opacity-70 flex items-center justify-center">
									<div className="border border-red-500 w-[160px] h-[160px] opacity-70 flex items-center justify-center">
										<div className="border border-red-500 w-[100px] h-[100px] opacity-70"></div>
									</div>
								</div>
								<div className="absolute w-full h-[2px] bg-red-500/50"></div>
								<div className="absolute w-[2px] h-full bg-red-500/50"></div>
							</div>

							{/* Status indicators */}
							<div className="absolute top-2 left-2 text-xs font-mono text-blue-400 bg-black/50 px-2 py-1">
								REC • LIVE
							</div>
							<div className="absolute top-2 right-2 text-xs font-mono text-blue-400 bg-black/50 px-2 py-1">
								{new Date().toLocaleTimeString()}
							</div>
							<div className="absolute bottom-2 left-2 text-xs font-mono text-blue-400 bg-black/50 px-2 py-1">
								FACIAL SCAN READY
							</div>
							<div className="absolute bottom-2 right-2 text-xs font-mono text-red-400 bg-black/50 px-2 py-1 flex items-center">
								<div className="h-2 w-2 rounded-full bg-red-500 mr-1 animate-pulse"></div>
								RECORDING
							</div>
						</div>
					</>
				) : (
					<div className="w-full h-[300px] bg-gray-900 flex flex-col items-center justify-center">
						{error ? (
							<div className="text-center">
								<AlertCircle className="h-8 w-8 text-red-500 mx-auto mb-2" />
								<p className="text-red-500 font-mono text-sm">{error}</p>
							</div>
						) : (
							<div className="text-center">
								<Camera className="h-8 w-8 text-blue-500 mx-auto mb-2" />
								<p className="text-blue-500 font-mono text-sm">SURVEILLANCE FEED INACTIVE</p>
							</div>
						)}
					</div>
				)}
			</div>

			{/* Hidden canvas for image capture */}
			<canvas ref={canvasRef} className="hidden" />

			<div className="flex gap-2">
				{!isCameraActive ? (
					<Button onClick={startCamera} className="flex-1 bg-blue-900 hover:bg-blue-800 font-mono">
						<Camera className="mr-2 h-4 w-4" />
						ACTIVATE SURVEILLANCE
					</Button>
				) : (
					<>
						<Button onClick={captureImage} variant="default" className="flex-1 bg-red-900 hover:bg-red-800 font-mono">
							<Camera className="mr-2 h-4 w-4" />
							CAPTURE SUBJECT
						</Button>
						<Button onClick={stopCamera} variant="outline" className="border-blue-800 text-blue-400 font-mono">
							<RefreshCw className="h-4 w-4" />
						</Button>
					</>
				)}
			</div>
		</div>
	)
}
