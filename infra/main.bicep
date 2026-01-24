// Main Bicep template for FootballAnalytics infrastructure
// Deploys: Container App, PostgreSQL, Key Vault, Container Registry, Azure Functions

targetScope = 'resourceGroup'

@description('Environment name')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'prod'

@description('Location for all resources')
param location string = 'uksouth'

@description('PostgreSQL administrator password')
@secure()
param postgresPassword string

@description('Football Data API key')
@secure()
param footballDataApiKey string

@description('The Odds API key')
@secure()
param oddsApiKey string

@description('Anthropic API key')
@secure()
param anthropicApiKey string

// Variables
var prefix = 'fa'
var uniqueSuffix = uniqueString(resourceGroup().id)
var tags = {
  environment: environment
  application: 'footballanalytics'
}

// Container Registry
module acr 'modules/container-registry.bicep' = {
  name: 'acr-deployment'
  params: {
    name: '${prefix}acr${uniqueSuffix}'
    location: location
    tags: tags
  }
}

// Key Vault
module keyVault 'modules/keyvault.bicep' = {
  name: 'keyvault-deployment'
  params: {
    name: '${prefix}-kv-${uniqueSuffix}'
    location: location
    tags: tags
    secrets: [
      {
        name: 'postgres-password'
        value: postgresPassword
      }
      {
        name: 'football-data-api-key'
        value: footballDataApiKey
      }
      {
        name: 'odds-api-key'
        value: oddsApiKey
      }
      {
        name: 'anthropic-api-key'
        value: anthropicApiKey
      }
    ]
  }
}

// PostgreSQL Flexible Server
module postgres 'modules/postgres.bicep' = {
  name: 'postgres-deployment'
  params: {
    name: '${prefix}-postgres-${uniqueSuffix}'
    location: location
    tags: tags
    administratorPassword: postgresPassword
    databaseName: 'football'
  }
}

// Container Apps Environment and API
module containerApp 'modules/container-app.bicep' = {
  name: 'container-app-deployment'
  params: {
    name: '${prefix}-api-${environment}'
    location: location
    tags: tags
    containerRegistryName: acr.outputs.name
    containerImage: '${acr.outputs.loginServer}/footballanalytics-api:latest'
    keyVaultName: keyVault.outputs.name
    postgresHost: postgres.outputs.fqdn
    postgresDatabase: 'football'
    postgresUser: 'faadmin'
  }
  dependsOn: [
    acr
    keyVault
    postgres
  ]
}

// Azure Functions for scheduled jobs
module functions 'modules/functions.bicep' = {
  name: 'functions-deployment'
  params: {
    name: '${prefix}-func-${environment}'
    location: location
    tags: tags
    keyVaultName: keyVault.outputs.name
    postgresConnectionString: 'postgresql://faadmin:${postgresPassword}@${postgres.outputs.fqdn}:5432/football'
  }
  dependsOn: [
    keyVault
    postgres
  ]
}

// Outputs
output containerAppUrl string = containerApp.outputs.url
output containerRegistryLoginServer string = acr.outputs.loginServer
output postgresHost string = postgres.outputs.fqdn
output keyVaultName string = keyVault.outputs.name
output functionsAppName string = functions.outputs.name
