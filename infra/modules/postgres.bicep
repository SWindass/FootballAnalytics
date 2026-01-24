// Azure PostgreSQL Flexible Server module

@description('Name of the PostgreSQL server')
param name string

@description('Location for the server')
param location string

@description('Tags to apply')
param tags object

@description('Administrator password')
@secure()
param administratorPassword string

@description('Database name')
param databaseName string = 'football'

resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2023-03-01-preview' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: 'Standard_B1ms'  // Burstable, 1 vCore, 2GB RAM - ~$15-20/month
    tier: 'Burstable'
  }
  properties: {
    version: '16'
    administratorLogin: 'faadmin'
    administratorLoginPassword: administratorPassword
    storage: {
      storageSizeGB: 32
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Disabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
  }
}

// Firewall rule to allow Azure services
resource firewallRule 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2023-03-01-preview' = {
  parent: postgres
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// Create database
resource database 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2023-03-01-preview' = {
  parent: postgres
  name: databaseName
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

output name string = postgres.name
output fqdn string = postgres.properties.fullyQualifiedDomainName
output id string = postgres.id
