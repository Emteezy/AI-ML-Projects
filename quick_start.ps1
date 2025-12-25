# Quick Start Script for Windows
# Run this to start all portfolio projects

Write-Host "================================" -ForegroundColor Cyan
Write-Host "AI/ML Portfolio - Quick Start" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker..." -ForegroundColor Yellow
$dockerRunning = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}
Write-Host "✓ Docker is running" -ForegroundColor Green
Write-Host ""

# Start services
Write-Host "Starting all services..." -ForegroundColor Yellow
docker-compose up -d

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Services started successfully!" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "Waiting for services to be ready (30 seconds)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
    
    # Seed CMMS database
    Write-Host ""
    Write-Host "Seeding CMMS database..." -ForegroundColor Yellow
    docker-compose exec -T cmms_api python seed_data.py
    
    Write-Host ""
    Write-Host "================================" -ForegroundColor Green
    Write-Host "✓ ALL SERVICES READY!" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Access your applications at:" -ForegroundColor Cyan
    Write-Host "  • Automation HMI:    http://localhost:8501" -ForegroundColor White
    Write-Host "  • Automation API:    http://localhost:8001/docs" -ForegroundColor White
    Write-Host "  • CMMS Dashboard:    http://localhost:8502" -ForegroundColor White
    Write-Host "  • CMMS API:          http://localhost:8003/docs" -ForegroundColor White
    Write-Host ""
    Write-Host "To stop services: docker-compose down" -ForegroundColor Yellow
    Write-Host "To view logs:     docker-compose logs -f" -ForegroundColor Yellow
    Write-Host ""
    
    # Open browsers
    $openBrowser = Read-Host "Open browsers automatically? (Y/n)"
    if ($openBrowser -eq "" -or $openBrowser -eq "Y" -or $openBrowser -eq "y") {
        Start-Process "http://localhost:8501"
        Start-Sleep -Seconds 2
        Start-Process "http://localhost:8502"
    }
    
} else {
    Write-Host "ERROR: Failed to start services!" -ForegroundColor Red
    Write-Host "Check docker-compose logs for details" -ForegroundColor Red
    exit 1
}

