export default function Header() {
	return (
		<header className="bg-blue-950 border-b-2 border-blue-800">
			<div className="container mx-auto py-4 px-4">
				<div className="flex items-center justify-between">
					<div className="flex items-center">
						<div className="mr-4">
							<div className="font-bold text-2xl tracking-wider text-white">SAURON OS</div>
							<div className="text-xs text-blue-400 font-mono">Facial Detabase</div>
						</div>
						<div className="h-8 w-[1px] bg-blue-800 mx-4 hidden sm:block"></div>
						<div className="hidden sm:block">
							<div className="text-sm font-bold text-white">FACIAL RECOGNITION DATABASE</div>
							<div className="text-xs text-blue-400 font-mono">CLASSIFIED • TOP SECRET</div>
						</div>
					</div>

					<div className="text-right">
						<div className="text-xs text-blue-400 font-mono">SECURITY LEVEL: ALPHA</div>
						<div className="text-xs text-blue-400 font-mono flex items-center justify-end">
							<div className="h-1.5 w-1.5 rounded-full bg-green-500 mr-1 animate-pulse"></div>
							SECURE CONNECTION
						</div>
					</div>
				</div>
			</div>
		</header>
	)
}
